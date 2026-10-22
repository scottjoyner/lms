# Joyner R9700 RAM procurement — exact-SKU quote form (October 22, 2026)

**Purpose:** independently review an exact supplier quote for a matched **2 × 32 GiB DDR5 UDIMM** kit before seeking operator approval. This is a *form validator only*; it neither authenticates the vendor nor verifies warehouse inventory. No automatic purchase, purchase approval, BIOS upgrade, fleet maintenance or IQ3_S inference permission can ever result.

## Current sources and decision

- Joyner: ASUS PRIME B650M-A AX II + Ryzen 5 7600X + BIOS 3014.
- Physical Oct 22 unprivileged checks: ~14.75 GiB OS-visible RAM, ~4.14 GiB available, almost all swap occupied. **Installed DIMM Type-17 slots unverified**; authorized local-console collection still necessary.
- The Oct 22 direct ASUS Ryzen-7000 2×32GB QVL refresh returned 104 rows and normalized SHA-256 `a646aabbb24c2c13e66db62afdd132e13cbdd4232adcc01ea2019adac9676fef`, identical to Oct 21. Script pins six directly listed JEDEC 1.1V Team Group kit SKUs: `TED564G5200C42DC01`, `TPBD564G5200HC42DC01`, `TPSD564G5200HC42DC01` (5200); `TED564G4800C40DC01`, `TPBD564G4800HC40DC01`, `TPSD564G4800HC40DC01` (4800).
- Manufacturer's native 5200 32GBx2 Elite kit `TED564G5200C42DC01` is shown in TEAMGROUP's official datasheet (CL42, 1.1 V): https://images.teamgroupinc.com/products/memory/u-dimm/ddr5/elite/edm/elite-en.pdf
- A public vendor brand catalog **lists** the exact preferred SKU, with **member-only pricing**: https://shop.perfectcloudsolutions.com/brands/team/ . The more specific previously captured supplier listing named item 2283915, but the current public web check could not retrieve a live checkout price. Walmart lists the same SKU *out of stock*: https://www.walmart.com/ip/3666966749 . **No actionable delivered price, verified allocatable warehouse quantity, current quote reference or operator order authority has been established.**
- Supplier catalog, quote contents, user-provided timestamps and static QVL reference are all **untrusted inputs**, even when well formed. The quote's self-asserted operator-confirmation flags are not an independent attestation.
- Prior research issue [LMS #53](https://github.com/scottjoyner/lms/issues/53) tracks physical DDR5 procurement. [LMS #51](https://github.com/scottjoyner/lms/issues/51) tracks the verified IQ3_S weight/pack hashes and pending inference.

## How to run without touching production

The script uses **only local JSON**; there are **no network calls, vendor contacts, email, process inspection, privileged operations, purchase API calls or file writes**.

```bash
python3 scripts/r9700_ram_quote_review.py
# Expected: HOLD / NO_OPERATOR_SUPPLIED_SUPPLIER_QUOTE, exit code 2.

python3 scripts/r9700_ram_quote_review.py --quote-file /absolute/path/operator-redacted-supplier-quote.json
# For a completely syntactically valid mock form: FORM_COMPLETE_REVIEW_REQUIRED,
# but STILL exit code 2, with memory_purchase_authorized=false.
```

A real operator-approved redacted quote JSON should have the following exact keys and types (shown with **blank or invalid placeholders, NOT a real quote**):

```json
{
  "schema": "r9700-memory-supplier-quote.v1",
  "seller_name": "",
  "seller_website": "",
  "seller_quote_reference": "",
  "seller_contact_verified_by_operator": false,
  "manufacturer_part_number": "TED564G5200C42DC01",
  "manufacturer": "TEAMGROUP",
  "module_count": 2,
  "capacity_per_module_gib": 32,
  "form_factor": "288-pin DDR5 UDIMM NON-ECC UNBUFFERED",
  "initial_speed_mt_s": 5200,
  "voltage_v": "1.1",
  "manufacturer_revision_or_lot": "",
  "quote_issued_utc": "",
  "quote_expires_utc": "",
  "claimed_stock_checked_utc": "",
  "claimed_stock_status": "",
  "ship_to_country": "US",
  "currency": "USD",
  "item_usd": "",
  "shipping_usd": "",
  "sales_tax_usd": "",
  "grand_total_usd": "",
  "sealed_new_kit": false,
  "return_policy_reviewed_by_operator": false
}
```

The operator must fill fields only after checking a real vendor quote, independently confirming the supplier identity and exact SKU/lot. **Do not paste names, home shipping address, card details, payment links, invoice secrets or credentials into GitHub.** The parser refuses unknown fields (including addresses) and reads no more than 16 KiB from a regular non-symlink file.

A form is marked `FORM_COMPLETE_REVIEW_REQUIRED` only when all fields are present, match the pinned JEDEC model geometry, use HTTPS, have 48-hour stock-claim freshness, a quote not older than 7 days or expired, properly calculated USD item + shipping + tax = grand total, and claim factory sealed stock. **This status proves form consistency only.** It does *not* verify invoice authenticity, supplier custody, live stock, validity of the operator checkmarks, QVL signing provenance, board BIOS/lot compatibility, supplier trustworthiness or independent physical DIMM population.

## Physical gating after quote review

**Mandatory separate operator approvals:** current privileged redacted four-slot inventory; exact seller quote and revision/lot + return policy; budget / price approval; maintenance window; physically install a qualified matched 2×32 GiB set in A2/B2; post-boot RAM/firmware compatibility, sustained memory stability, ≥60 GiB OS-visible RAM and ≥8 GiB available RAM; separately witness GPU exclusive custody and benchmark quality/TTFT/SSD I/O; require proven owner-bound routing receipts before model admission.

The tool always returns `production_admission="NO_GO"` and `model_launch_allowed=false`. Neither a passing form nor a merchant stock badge can move these gates.