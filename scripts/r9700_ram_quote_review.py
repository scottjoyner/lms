#!/usr/bin/env python3
"""Offline, non-authorizing quote-form audit for Joyner's RAM upgrade.

Catalog and vendor fields are untrusted. A passing FORM check cannot
authenticate a supplier, prove availability, authorize a purchase or
permit hardware installation / model inference. No network, email, checkout,
firmware, root access, or fleet actions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import urlsplit

_MAX_INPUT_BYTES = 16 * 1024
_BASE_QVL_SHA256 = "a646aabbb24c2c13e66db62afdd132e13cbdd4232adcc01ea2019adac9676fef"
# Directly transcribed from the 2026-10-22 exact-board ASUS Ryzen 7000 2x32 GiB QVL.
# This mapping is a LOCAL reference, not a signed ASUS attestation or BIOS-3014 proof.
_QVL_JEDEC_KITS = {
    "TED564G5200C42DC01": 5200,
    "TPBD564G5200HC42DC01": 5200,
    "TPSD564G5200HC42DC01": 5200,
    "TED564G4800C40DC01": 4800,
    "TPBD564G4800HC40DC01": 4800,
    "TPSD564G4800HC40DC01": 4800,
}
_REQUIRED = {
    "schema", "seller_name", "seller_website", "seller_quote_reference",
    "seller_contact_verified_by_operator", "manufacturer_part_number",
    "manufacturer", "module_count", "capacity_per_module_gib",
    "form_factor", "initial_speed_mt_s", "voltage_v",
    "manufacturer_revision_or_lot", "quote_issued_utc", "quote_expires_utc",
    "claimed_stock_checked_utc", "claimed_stock_status", "ship_to_country",
    "currency", "item_usd", "shipping_usd", "sales_tax_usd",
    "grand_total_usd", "sealed_new_kit", "return_policy_reviewed_by_operator",
}
_MONEY = re.compile(r"(?:0|[1-9][0-9]{0,4})\.[0-9]{2}\Z")
_SAFE_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+/-]{2,95}\Z")
_SAFE_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9 &.,()+/:-]{2,99}\Z")


def _when(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or len(raw) > 40:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _money(raw: Any) -> Decimal | None:
    if not isinstance(raw, str) or not _MONEY.fullmatch(raw):
        return None
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _domain(raw: Any) -> str | None:
    if not isinstance(raw, str) or len(raw) > 256:
        return None
    try:
        url = urlsplit(raw)
        if (url.scheme != "https" or not url.hostname or url.username
                or url.password or url.port not in (None, 443)
                or not url.hostname.endswith((".com", ".net", ".org", ".us"))):
            return None
        return url.hostname.lower()
    except ValueError:
        return None


def review(quote: dict[str, Any] | None, *, now: datetime | None = None) -> dict:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    reasons: list[str] = []
    if quote is None:
        quote = {}
        reasons.append("NO_OPERATOR_SUPPLIED_SUPPLIER_QUOTE")
    elif not isinstance(quote, dict):
        reasons.append("QUOTE_MUST_BE_JSON_OBJECT")
        quote = {}
    missing = sorted(_REQUIRED - quote.keys())
    if quote and missing:
        reasons.append("MISSING_FIELDS:" + ",".join(missing))
    if quote and set(quote) != _REQUIRED:
        reasons.append("UNEXPECTED_FIELDS_OR_MISSING_FIELDS")
    if quote.get("schema") != "r9700-memory-supplier-quote.v1":
        if quote:
            reasons.append("QUOTE_SCHEMA_MISMATCH")
    part = quote.get("manufacturer_part_number")
    expected_speed = _QVL_JEDEC_KITS.get(part) if isinstance(part, str) else None
    if expected_speed is None and quote:
        reasons.append("EXACT_SKU_NOT_IN_PINNED_JEDEC_QVL_SHORTLIST")
    if quote:
        shape = (
            quote.get("manufacturer") == "TEAMGROUP"
            and type(quote.get("module_count")) is int
            and quote.get("module_count") == 2
            and type(quote.get("capacity_per_module_gib")) is int
            and quote.get("capacity_per_module_gib") == 32
            and quote.get("form_factor") == "288-pin DDR5 UDIMM NON-ECC UNBUFFERED"
            and type(quote.get("initial_speed_mt_s")) is int
            and quote.get("initial_speed_mt_s") == expected_speed
            and quote.get("voltage_v") == "1.1"
        )
        if not shape:
            reasons.append("MEMORY_GEOMETRY_OR_JEDEC_SPEC_NOT_VERIFIED")
        if (not isinstance(quote.get("seller_name"), str)
                or not _SAFE_TEXT.fullmatch(quote["seller_name"])):
            reasons.append("SELLER_LEGAL_NAME_FORMAT_INVALID")
        domain = _domain(quote.get("seller_website"))
        if not domain:
            reasons.append("SELLER_HTTPS_DOMAIN_NOT_VERIFIABLE_FROM_FORM")
        for field in ("seller_quote_reference", "manufacturer_revision_or_lot"):
            if (not isinstance(quote.get(field), str)
                    or not _SAFE_CODE.fullmatch(quote[field])):
                reasons.append("MISSING_OR_INVALID_" + field.upper())
        if quote.get("claimed_stock_status") != "SUPPLIER_CLAIMS_ALLOCATABLE_1_KIT":
            reasons.append("STOCK_NOT_CLAIMED_ALLOCATABLE_FOR_ONE_KIT")
        if quote.get("ship_to_country") != "US" or quote.get("currency") != "USD":
            reasons.append("US_DELIVERY_AND_CURRENCY_NOT_CONFIRMED")
        for flag in ("seller_contact_verified_by_operator", "sealed_new_kit",
                     "return_policy_reviewed_by_operator"):
            if quote.get(flag) is not True:
                reasons.append("OPERATOR_REVIEW_OR_NEW_KIT_UNVERIFIED_" + flag.upper())
        issued, expires, checked = (
            _when(quote.get(k))
            for k in ("quote_issued_utc", "quote_expires_utc", "claimed_stock_checked_utc")
        )
        if (not all((issued, expires, checked)) or issued > now + timedelta(minutes=10)
                or checked > now + timedelta(minutes=10)
                or issued < now - timedelta(days=7)
                or checked < now - timedelta(hours=48)
                or checked < issued - timedelta(hours=24)
                or expires <= now
                or expires > issued + timedelta(days=7)):
            reasons.append("QUOTE_OR_STOCK_AGE_WINDOW_UNVERIFIED")
        charges = [_money(quote.get(k)) for k in (
            "item_usd", "shipping_usd", "sales_tax_usd", "grand_total_usd",
        )]
        if (any(x is None for x in charges) or charges[0] <= 0
                or charges[-1] <= 0 or charges[-1] > Decimal("5000.00")
                or sum(charges[:3]) != charges[3]):
            reasons.append("QUOTE_DOLLARS_INVALID_OR_COMPONENTS_DO_NOT_ADD_UP")
    else:
        domain = None
    return {
        "schema": "r9700-ram-quote-review.v1",
        "status": "FORM_COMPLETE_REVIEW_REQUIRED" if not reasons else "HOLD",
        "sku": part if expected_speed is not None else None,
        "seller_domain_for_manual_followup": domain,
        "source_qvl_normalized_sha256_reference": _BASE_QVL_SHA256,
        "source_qvl_signer_custody_verified": False,
        "supplier_identity_or_quote_authenticity_verified": False,
        "supplier_warehouse_stock_physically_verified": False,
        "quote_validity_reasons": reasons,
        "manual_gates_remaining": [
            "INDEPENDENT_SUPPLIER_QUOTE_ORIGIN_AND_STOCK_CONFIRMATION",
            "OPERATOR_PRIVILEGED_DIMM_SLOT_INVENTORY",
            "EXACT_MODULE_REVISION_LOT_AND_BIOS_3014_REVIEW",
            "EXPLICIT_OPERATOR_PURCHASE_APPROVAL",
            "OPERATOR_PHYSICAL_INSTALL_AND_MEMORY_STABILITY",
            "GPU_MAINTENANCE_AND_IQ3_INFERENCE_GATES",
        ],
        "memory_purchase_authorized": False,
        "memory_install_authorized": False,
        "model_launch_allowed": False,
        "production_admission": "NO_GO",
    }


def _read_quote(path: Path) -> dict:
    """Refuse symlink, nonregular, or oversized documents; parse bounded JSON."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        try:
            meta = os.fstat(fd)
            if not stat.S_ISREG(meta.st_mode) or not 1 <= meta.st_size <= _MAX_INPUT_BYTES:
                raise ValueError("quote must be a regular 1..16384-byte JSON file")
            data = os.read(fd, _MAX_INPUT_BYTES + 1)
        finally:
            os.close(fd)
    except OSError as exc:
        raise ValueError("quote is unreadable or unsafe") from exc
    if len(data) > _MAX_INPUT_BYTES:
        raise ValueError("quote size limit exceeded")
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("quote is not valid UTF-8 JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError("quote must be a JSON object")
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quote-file", type=Path)
    args = parser.parse_args(argv)
    try:
        quote = _read_quote(args.quote_file) if args.quote_file else None
        result = review(quote)
    except ValueError:
        result = review(None)
        result["quote_validity_reasons"] = ["QUOTE_FILE_UNSAFE_OR_INVALID"]
    print(json.dumps(result, indent=2, sort_keys=True))
    # Form completeness never becomes a purchase/installation authorization.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())