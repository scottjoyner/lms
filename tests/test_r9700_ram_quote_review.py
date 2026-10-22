"""Adversarial offline quote review tests; all outcomes remain NO_GO."""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "scripts/r9700_ram_quote_review.py"
_spec = importlib.util.spec_from_file_location("r9700_ram_quote_review", _MODULE)
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

NOW = datetime(2026, 10, 22, 16, 45, tzinfo=timezone.utc)


def iso(hours=0):
    return (NOW + timedelta(hours=hours)).isoformat()


def valid_quote(**replacements):
    data = {
        "schema": "r9700-memory-supplier-quote.v1",
        "seller_name": "Perfect Cloud Solutions",
        "seller_website": "https://shop.perfectcloudsolutions.com/products/2283915",
        "seller_quote_reference": "SYNTHETIC-TEST-ONLY",
        "seller_contact_verified_by_operator": True,
        "manufacturer_part_number": "TED564G5200C42DC01",
        "manufacturer": "TEAMGROUP",
        "module_count": 2,
        "capacity_per_module_gib": 32,
        "form_factor": "288-pin DDR5 UDIMM NON-ECC UNBUFFERED",
        "initial_speed_mt_s": 5200,
        "voltage_v": "1.1",
        "manufacturer_revision_or_lot": "SYNTHETIC-LOT",
        "quote_issued_utc": iso(-1),
        "quote_expires_utc": iso(48),
        "claimed_stock_checked_utc": iso(-1),
        "claimed_stock_status": "SUPPLIER_CLAIMS_ALLOCATABLE_1_KIT",
        "ship_to_country": "US",
        "currency": "USD",
        "item_usd": "201.00",
        "shipping_usd": "10.00",
        "sales_tax_usd": "11.00",
        "grand_total_usd": "222.00",
        "sealed_new_kit": True,
        "return_policy_reviewed_by_operator": True,
    }
    data.update(replacements)
    return data


def outcome(data=None):
    return gate.review(data, now=NOW)


def assert_no_go(row):
    assert row["memory_purchase_authorized"] is False
    assert row["memory_install_authorized"] is False
    assert row["model_launch_allowed"] is False
    assert row["production_admission"] == "NO_GO"
    assert row["supplier_warehouse_stock_physically_verified"] is False
    assert row["supplier_identity_or_quote_authenticity_verified"] is False
    assert row["source_qvl_signer_custody_verified"] is False


def test_no_quote_is_safe_and_incomplete():
    result = outcome()
    assert result["status"] == "HOLD"
    assert "NO_OPERATOR_SUPPLIED_SUPPLIER_QUOTE" in result["quote_validity_reasons"]
    assert_no_go(result)


def test_complete_synthetic_form_is_never_purchase_approval():
    result = outcome(valid_quote())
    assert result["status"] == "FORM_COMPLETE_REVIEW_REQUIRED"
    assert result["sku"] == "TED564G5200C42DC01"
    assert result["seller_domain_for_manual_followup"] == "shop.perfectcloudsolutions.com"
    assert result["quote_validity_reasons"] == []
    assert_no_go(result)


@pytest.mark.parametrize("sku,speed", [
    ("TED564G5200C42DC01", 5200),
    ("TPBD564G5200HC42DC01", 5200),
    ("TPSD564G5200HC42DC01", 5200),
    ("TED564G4800C40DC01", 4800),
    ("TPBD564G4800HC40DC01", 4800),
    ("TPSD564G4800HC40DC01", 4800),
])
def test_known_asus_qvl_skus_only_form_review(sku, speed):
    result = outcome(valid_quote(manufacturer_part_number=sku, initial_speed_mt_s=speed))
    assert result["status"] == "FORM_COMPLETE_REVIEW_REQUIRED"
    assert_no_go(result)


@pytest.mark.parametrize("sku", [
    "TED564G5200C42DC02", "TED564G5200C42DC01-REPLACEMENT",
    "TED564G5200C42DC0I", "KF548C38BBK2-64",
    "TPBD564G4800HC40DC0l", "TEAMGROUP DDR4",
])
def test_wrong_or_unqualified_sku_rejected(sku):
    result = outcome(valid_quote(manufacturer_part_number=sku))
    assert result["status"] == "HOLD"
    assert "EXACT_SKU_NOT_IN_PINNED_JEDEC_QVL_SHORTLIST" in result["quote_validity_reasons"]
    assert result["sku"] is None
    assert_no_go(result)


@pytest.mark.parametrize("edit", [
    {"module_count": 1}, {"module_count": True},
    {"capacity_per_module_gib": 16}, {"capacity_per_module_gib": 64},
    {"capacity_per_module_gib": "32"},
    {"form_factor": "262-pin DDR5 SO-DIMM"},
    {"form_factor": "288-pin DDR5 RDIMM"},
    {"form_factor": "288-pin DDR4 UDIMM"},
    {"initial_speed_mt_s": 6000},
    {"initial_speed_mt_s": 4800},
    {"voltage_v": 1.1}, {"voltage_v": "1.25"},
    {"manufacturer": "UNKNOWN"},
])
def test_form_factor_capacity_speed_voltage_rejected(edit):
    row = outcome(valid_quote(**edit))
    assert row["status"] == "HOLD"
    assert "MEMORY_GEOMETRY_OR_JEDEC_SPEC_NOT_VERIFIED" in row["quote_validity_reasons"]
    assert_no_go(row)


@pytest.mark.parametrize("change", [
    {"quote_issued_utc": iso(-192)},
    {"quote_issued_utc": iso(2)},
    {"quote_issued_utc": "2026-10-22"},
    {"quote_expires_utc": iso(-1)},
    {"quote_expires_utc": iso(168)},
    {"claimed_stock_checked_utc": iso(-50)},
    {"claimed_stock_checked_utc": iso(3)},
    {"claimed_stock_checked_utc": iso(-36), "quote_issued_utc": iso(-1)},
])
def test_stale_future_or_expired_quote_rejected(change):
    row = outcome(valid_quote(**change))
    assert row["status"] == "HOLD"
    assert "QUOTE_OR_STOCK_AGE_WINDOW_UNVERIFIED" in row["quote_validity_reasons"]
    assert_no_go(row)


@pytest.mark.parametrize("change", [
    {"item_usd": 201.00}, {"item_usd": "-1.00"},
    {"item_usd": "1e3"}, {"grand_total_usd": "221.99"},
    {"grand_total_usd": "6000.00"},
    {"sales_tax_usd": True}, {"shipping_usd": "$0.00"},
    {"item_usd": "0.00"}, {"item_usd": "Inf"},
])
def test_wrong_currency_or_malformed_total_rejected(change):
    row = outcome(valid_quote(**change))
    assert row["status"] == "HOLD"
    assert "QUOTE_DOLLARS_INVALID_OR_COMPONENTS_DO_NOT_ADD_UP" in row["quote_validity_reasons"]
    assert_no_go(row)


@pytest.mark.parametrize("change", [
    {"seller_website": "http://shop.perfectcloudsolutions.com/a"},
    {"seller_website": "https://someone:password@shop.perfectcloudsolutions.com/a"},
    {"seller_website": "https://shop.perfectcloudsolutions.com:8443/a"},
    {"seller_website": "https://localhost:443/a"},
    {"seller_name": ""},
    {"seller_quote_reference": "QUOTE\\nUNSAFE"},
    {"manufacturer_revision_or_lot": "UNKNOWN LOT!"},
])
def test_invalid_seller_identifiers_rejected(change):
    row = outcome(valid_quote(**change))
    assert row["status"] == "HOLD"
    assert_no_go(row)


@pytest.mark.parametrize("change", [
    {"claimed_stock_status": "IN_STOCK_SEARCH_SNIPPET"},
    {"seller_contact_verified_by_operator": False},
    {"return_policy_reviewed_by_operator": False},
    {"sealed_new_kit": False},
    {"ship_to_country": "CA"}, {"currency": "CAD"},
])
def test_unverified_stock_account_or_delivery_rejected(change):
    row = outcome(valid_quote(**change))
    assert row["status"] == "HOLD"
    assert_no_go(row)


def test_unknown_fields_and_missing_fields_rejected():
    quote = valid_quote()
    quote["destination_address"] = "do not store"
    result = outcome(quote)
    assert result["status"] == "HOLD"
    assert "UNEXPECTED_FIELDS_OR_MISSING_FIELDS" in result["quote_validity_reasons"]
    assert_no_go(result)
    quote.pop("destination_address")
    quote.pop("manufacturer_revision_or_lot")
    result = outcome(quote)
    assert result["status"] == "HOLD"
    assert any(x.startswith("MISSING_FIELDS") for x in result["quote_validity_reasons"])
    assert_no_go(result)


def test_input_reader_rejects_symlink(tmp_path):
    p = tmp_path / "quote.json"
    p.write_text(json.dumps(valid_quote()))
    link = tmp_path / "alias"
    link.symlink_to(p)
    with pytest.raises(ValueError):
        gate._read_quote(link)


@pytest.mark.parametrize("invalid", [b"", b"x" * 16385, b"not json", b"[]"])
def test_input_reader_rejects_bad_json_and_sizes(tmp_path, invalid):
    path = tmp_path / "quote.json"
    path.write_bytes(invalid)
    with pytest.raises(ValueError):
        gate._read_quote(path)


def test_input_reader_accepts_bounded_regular_json(tmp_path):
    p = tmp_path / "quote.json"
    p.write_text(json.dumps(valid_quote()))
    assert gate._read_quote(p)["manufacturer_part_number"] == "TED564G5200C42DC01"


def test_cli_always_exit_nonzero_even_with_form_complete(tmp_path, capsys):
    p = tmp_path / "quote.json"
    p.write_text(json.dumps(valid_quote()))
    rc = gate.main(["--quote-file", str(p)])
    assert rc == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "HOLD" or result["status"] == "FORM_COMPLETE_REVIEW_REQUIRED"
    assert_no_go(result)


def test_now_no_quote_cli_is_hold(capsys):
    rc = gate.main([])
    assert rc == 2
    assert_no_go(json.loads(capsys.readouterr().out))