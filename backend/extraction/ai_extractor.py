"""
AI Extractor Module
===================
Extracts structured financial data from document text using Groq AI API.

This module uses Groq's Llama 3 model for intelligent extraction of:
- Invoice/Bill numbers
- Dates
- Vendor/Supplier names
- Line items (item, quantity, rate, amount)
- Subtotal, Tax, Total

All processing is done in-memory. No data is stored or logged.
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Load environment variables from backend/.env regardless of launch directory
from dotenv import load_dotenv

_BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_BACKEND_ENV_PATH)

# Groq AI client
from groq import Groq

logger = logging.getLogger("invoice_processing")

from .preprocessing import preprocess as adaptive_preprocess


@dataclass
class LineItem:
    """
    Represents a single line item from an invoice/bill.
    
    Attributes:
        item_name: Description of the item/product
        quantity: Number of units
        rate: Price per unit (defaults to 0.0 if not found)
        discount_percent: Discount percentage on this line item (e.g., 50 for 50%)
        amount: Total line amount AFTER discount (defaults to 0.0 if not found)
    """
    item_name: str
    quantity: float
    rate: float = 0.0
    discount_percent: float = 0.0
    amount: float = 0.0


@dataclass
class AdditionalCharge:
    """
    Represents an additional charge on an invoice (not a product).
    
    Examples: Packing charges, Freight, Shipping, Handling, Forwarding charges.
    These are NOT inventory items and should not affect stock balance.
    
    Attributes:
        charge_name: Name/description of the charge
        amount: Charge amount
    """
    charge_name: str
    amount: float = 0.0
    quantity: float = 0.0
    rate: float = 0.0


@dataclass
class ExtractedData:
    """
    Structured data extracted from a document.
    
    Attributes:
        invoice_number: Invoice or bill reference number
        date: Document date
        vendor_name: Seller / supplier (or company issuing a tax invoice)
        customer_name: Buyer / bill-to / consignee (sales); usually empty on pure purchase bills
        line_items: List of line items with quantity and pricing (products only)
        additional_charges: List of charges (packing, freight, etc.) - NOT products
        subtotal: Sum of line item amounts before tax
        cgst: Central GST amount
        sgst: State GST amount
        igst: Integrated GST amount (for inter-state)
        tax: Total tax amount (cgst + sgst or igst)
        total: Final total amount
        extraction_notes: Any notes about the extraction process
    """
    invoice_number: str = ""
    date: str = ""
    vendor_name: str = ""
    customer_name: str = ""
    line_items: List[LineItem] = field(default_factory=list)
    additional_charges: List[AdditionalCharge] = field(default_factory=list)
    subtotal: float = 0.0
    cgst: float = 0.0
    sgst: float = 0.0
    igst: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    extraction_notes: List[str] = field(default_factory=list)
    error_code: str = ""
    raw_llm_response: str = ""


def _split_trailing_discount_from_item_name(item_name: str, discount_percent: float) -> tuple:
    """
    If a discount like '50' or '50%' was merged into the product name (common PDF line-wrap issue),
    move it to discount_percent and return a clean item_name.
    """
    s = (item_name or "").strip()
    if not s:
        return s, discount_percent
    m = re.search(r"(\s+)(\d{1,2}(?:\.\d+)?)\s*%?\s*$", s)
    if not m:
        return s, discount_percent
    try:
        pct = float(m.group(2))
    except ValueError:
        return s, discount_percent
    if not (0 < pct <= 100):
        return s, discount_percent
    base = s[: m.start()].strip()
    if len(base) < 2:
        return s, discount_percent
    if discount_percent == 0:
        return base, pct
    if abs(discount_percent - pct) < 0.01:
        return base, discount_percent
    return s, discount_percent


def _render_tables_for_prompt(tables: Optional[list], max_tables: int = 3, max_rows: int = 25) -> str:
    """
    Convert extracted PDF tables into a compact TSV-like block for LLM consumption.
    tables may include entries like {"page": int, "data": DataFrame}.
    """
    if not tables:
        return ""

    blocks = []
    used = 0
    for t in tables:
        if used >= max_tables:
            break
        df = None
        page = None
        if isinstance(t, dict) and "data" in t:
            df = t.get("data")
            page = t.get("page")
        else:
            df = t
        if df is None:
            continue
        try:
            # DataFrame -> list of rows; keep as strings
            rows = df.fillna("").astype(str).values.tolist()
        except Exception:
            continue
        if not rows:
            continue
        rows = rows[:max_rows]
        header = f"--- TABLE (page={page}) ---" if page else "--- TABLE ---"
        lines = ["\t".join([c.strip() for c in row]) for row in rows]
        blocks.append("\n".join([header] + lines))
        used += 1

    return "\n\n".join(blocks).strip()


def _estimate_line_item_rows(text_content: str) -> int:
    """
    Heuristic: count likely table rows that start with a serial number (e.g. '1', '2', '3')
    near common headers. This is intentionally approximate for monitoring/extraction notes.
    """
    if not text_content:
        return 0
    lines = [ln.strip() for ln in text_content.splitlines() if ln.strip()]
    # Focus on the area after S.NO / ITEMS headers if present
    start_idx = 0
    for i, ln in enumerate(lines):
        if "s.no" in ln.lower() and ("items" in ln.lower() or "description" in ln.lower()):
            start_idx = i
            break
    cand = 0
    for ln in lines[start_idx:]:
        if re.match(r"^\d{1,3}\s+", ln):
            cand += 1
    return cand


def _safe_print(msg: str) -> None:
    """
    Never let stdout encoding issues break extraction.
    Windows terminals/services can throw UnicodeEncodeError for characters like '₹'.
    """
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            print(msg.encode("utf-8", "replace").decode("utf-8", "replace"))
        except Exception:
            # As a last resort, drop the message.
            pass


def _clean_text_for_llm(text: str, max_chars: int = 4000) -> str:
    """
    Token optimization: keep mostly the item-table region, normalize whitespace,
    remove obvious footer/header noise, de-duplicate repeated lines, and cap length.
    """
    if not text:
        return ""
    # Normalize whitespace
    text = text.replace("\t", " ")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    # Remove consecutive duplicates
    dedup = []
    last = None
    for ln in lines:
        if ln == last:
            continue
        dedup.append(ln)
        last = ln
    lines = dedup

    # Try to slice the likely line-items area: from S.NO header to TOTAL/Grand Total
    start = 0
    for i, ln in enumerate(lines):
        low = ln.lower()
        if ("s.no" in low or low.startswith("sl ")) and ("items" in low or "description" in low):
            start = i
            break
    end = len(lines)
    for i in range(start, len(lines)):
        low = lines[i].lower()
        if low.startswith("total") or "grand total" in low or low.startswith("received amount"):
            end = min(len(lines), i + 3)
            break

    core = lines[start:end] if end > start else lines

    # Drop obvious noise lines (bank details, terms) if they appear inside the slice
    noise_kw = (
        "bank details",
        "ifsc",
        "account no",
        "upi",
        "terms and conditions",
        "authorised signatory",
        "this is a computer generated",
        "declaration",
    )
    core = [ln for ln in core if not any(k in ln.lower() for k in noise_kw)]

    cleaned = "\n".join(core).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def _approx_tokens(s: str) -> int:
    # Rough heuristic (good enough for logging/monitoring)
    if not s:
        return 0
    return max(1, len(s) // 4)


def _safe_json_loads(raw: str) -> Optional[dict]:
    """
    Robust JSON parsing:
    - strip markdown fences
    - extract first {...} block
    - try a couple of minor repairs (trailing commas)
    """
    if not raw:
        return None
    txt = raw.strip()
    if txt.startswith("```"):
        parts = txt.split("```")
        if len(parts) >= 2:
            txt = parts[1]
        txt = txt.strip()
        if txt.startswith("json"):
            txt = txt[4:].strip()

    # Extract first JSON object
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = txt[start : end + 1]

    def _compute_simple_arithmetic(s: str) -> str:
        """
        Some LLM responses mistakenly include arithmetic expressions in numeric JSON fields,
        e.g. `"amount": 24194 - 21962.4` which is invalid JSON.
        This replaces simple `number op number` expressions with the computed literal number.
        """
        expr = re.compile(
            r"(?P<a>-?\d+(?:\.\d+)?)\s*(?P<op>[\+\-\*/])\s*(?P<b>-?\d+(?:\.\d+)?)"
        )

        def repl(m: re.Match) -> str:
            try:
                a = float(m.group("a"))
                b = float(m.group("b"))
                op = m.group("op")
                if op == "+":
                    v = a + b
                elif op == "-":
                    v = a - b
                elif op == "*":
                    v = a * b
                elif op == "/":
                    if b == 0:
                        return m.group(0)
                    v = a / b
                else:
                    return m.group(0)
                # Render as a JSON number (trim excessive trailing zeros)
                out = f"{v:.6f}".rstrip("0").rstrip(".")
                return out if out else "0"
            except Exception:
                return m.group(0)

        # Iterate a few times in case replacement reveals another expression
        for _ in range(3):
            new_s = expr.sub(repl, s)
            if new_s == s:
                break
            s = new_s
        return s

    for attempt in range(2):
        try:
            candidate = _compute_simple_arithmetic(candidate)
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Remove trailing commas before } or ]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return None


def _extract_header_fields(text: str) -> dict:
    """
    Regex-based header extraction (works even when LLM is rate-limited).
    Extracts invoice number, invoice date, vendor/seller, customer/billed-to, and totals when possible.
    """
    if not text:
        return {}
    t = text
    low = t.lower()

    out: dict = {}

    # Invoice number (multiple patterns)
    m = re.search(r"invoice\s*no\.?\s*[:.]?\s*([A-Za-z0-9\-\/]+)", t, flags=re.I)
    if m:
        out["invoice_number"] = m.group(1).strip()

    # Invoice date
    m = re.search(r"invoice\s*date\s*[:.]?\s*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})", t, flags=re.I)
    if not m:
        m = re.search(r"invoice\s*date\s*[:.]?\s*([0-9]{1,2}\-\w{3}\-[0-9]{2,4})", t, flags=re.I)
    if m:
        out["date"] = m.group(1).strip()

    # Total / amount after tax
    m = re.search(r"total\s*amount\s*after\s*tax\s*([0-9,]+\.\d{2})", t, flags=re.I)
    if not m:
        m = re.search(r"\bgrand\s*total\b.*?([0-9,]+\.\d{2})", t, flags=re.I)
    if m:
        try:
            out["total"] = float(m.group(1).replace(",", ""))
        except Exception:
            pass

    # Tax amounts (CGST/SGST/IGST)
    def _money(s: str) -> float:
        try:
            return float(s.replace(",", ""))
        except Exception:
            return 0.0

    m = re.search(r"\bcgst\b(?:\s*amount)?\s*[:.]?\s*([0-9,]+\.\d{2})", t, flags=re.I)
    if m:
        out["cgst"] = _money(m.group(1))
    m = re.search(r"\bsgst\b(?:\s*amount)?\s*[:.]?\s*([0-9,]+\.\d{2})", t, flags=re.I)
    if m:
        out["sgst"] = _money(m.group(1))
    m = re.search(r"\bigst\b(?:\s*amount)?\s*[:.]?\s*([0-9,]+\.\d{2})", t, flags=re.I)
    if m:
        out["igst"] = _money(m.group(1))

    # Customer / receiver (GST flattened): extract from the receiver block only (avoid "Contact Person Name")
    # "Billed to" is sometimes followed by the customer on the NEXT line in PDF text extraction.
    bt = re.search(r"\bbilled\s*to\s*:", t, flags=re.I)
    if bt:
        after = t[bt.end() : bt.end() + 400]
        # Try same-line capture first
        m = re.match(r"\s*([^\n]+)", after)
        cands: list[str] = []
        if m:
            cands.append(m.group(1).strip().strip(","))
        # Then try next few lines
        next_lines = [ln.strip() for ln in after.splitlines()[1:6] if ln.strip()]
        cands.extend(next_lines)

        for cand in cands:
            low = cand.lower()
            if (
                (not cand)
                or "#n/a" in low
                or "contact" in low
                or "details of" in low
                or "shipped to" in low
                or "gstin" in low
                or "state" in low
                or "name of product" in low
            ):
                continue
            # must look like a name/org (letters present)
            if not re.search(r"[A-Za-z]", cand):
                continue
            # pdf text sometimes duplicates the same name twice; collapse simple repeats
            words = cand.split()
            if len(words) >= 8 and len(words) % 2 == 0:
                half = len(words) // 2
                if words[:half] == words[half:]:
                    cand = " ".join(words[:half]).strip()
            # also handle simple repeated substring in the same line
            if len(cand) >= 20:
                mrep = re.match(r"^(.{8,}?)\\s+\\1$", cand)
                if mrep:
                    cand = mrep.group(1).strip()
            out["customer_name"] = cand
            break

    receiver_block = None
    mblk = re.search(r"details of receiver[\s\S]{0,1200}", t, flags=re.I)
    if mblk:
        receiver_block = mblk.group(0)
    if receiver_block:
        # Prefer "Name <customer>" inside this block
        for m in re.finditer(r"\bname\b\s+([^\n]+)", receiver_block, flags=re.I):
            cand = m.group(1).strip().strip(",")
            # Skip obvious non-values
            low = cand.lower()
            if (
                (not cand)
                or cand in ("0", "#N/A", "N/A", "Name")
                or "#n/a" in low
                or "contact" in low
                or "product" in low
                or "service" in low
                or "hsn" in low
            ):
                continue
            # Stop before "State" if present in same line
            cand = re.split(r"\bstate\b\s*:", cand, flags=re.I)[0].strip().strip(",")
            if cand:
                out["customer_name"] = cand
                break

    # Customer (clean sales): BILL TO next non-empty line
    m = re.search(r"\bBILL TO\b\s*\n([^\n]+)", t, flags=re.I)
    if m:
        out.setdefault("customer_name", m.group(1).strip())

    # Vendor/seller: common issuer line
    if "golden moment group" in low:
        out["vendor_name"] = "Golden Moment Group"

    return out


def _fallback_extract_line_items(text: str, max_items: int = 50) -> List[LineItem]:
    """
    Regex-based fallback extraction when LLM fails/returns empty.
    Attempts to detect rows like: '<name> <qty> PCS <rate> <amount>' or similar.
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items: List[LineItem] = []

    # Start after header if found
    start = 0
    for i, ln in enumerate(lines):
        low = ln.lower()
        if ("s.no" in low or re.match(r"^sl\b", low)) and ("items" in low or "description" in low):
            start = i + 1
            break

    # Ignore standalone discount lines like "(50%)"
    disc_only = re.compile(r"^\(?\s*\d{1,2}(?:\.\d+)?\s*%\s*\)?$")
    num = re.compile(r"^-?\d+(?:,\d{3})*(?:\.\d+)?$")
    charge_kw = ("freight", "courier", "p&f", "p & f", "packing", "forwarding", "charges", "gst", "cgst", "sgst", "igst", "round off")
    inline_disc = re.compile(r"\(\s*(\d{1,2}(?:\.\d+)?)\s*%\s*\)")
    unit_tokens = {"pcs", "pc", "no", "no.", "nos", "nos."}
    skip_prefix = ("total", "grand total", "received amount", "previous balance", "current balance", "tax amount", "amount after tax")
    skip_exact = {"total", "grand", "grandtotal", "total:"}

    percent_line_re = re.compile(r"^\s*(?:\(?\s*\d+(?:\.\d+)?\s*%\s*\)?\s*)+$")

    for idx, ln in enumerate(lines[start:], start=start):
        low_ln = ln.lower().strip()
        if any(low_ln.startswith(p) for p in skip_prefix):
            continue
        if disc_only.match(ln.replace(" ", "")):
            continue
        if any(k in ln.lower() for k in charge_kw):
            continue
        # STRICT DISCOUNT RULE:
        # Any value inside parentheses with % is ALWAYS discount.
        # Remove it before parsing numbers so it never pollutes qty/rate/amount.
        discount_percent = 0.0
        m_all = list(inline_disc.finditer(ln))
        if m_all:
            # If multiple, take the last (closest to amount in typical layouts)
            try:
                discount_percent = float(m_all[-1].group(1))
            except Exception:
                discount_percent = 0.0
            ln = inline_disc.sub(" ", ln)
            ln = re.sub(r"\s+", " ", ln).strip()

        # Some newer sales bills place discount in a tiny font on the next line.
        # Prefer that value over any number already present in the row.
        if idx + 1 < len(lines):
            next_ln = lines[idx + 1].strip()
            if percent_line_re.match(next_ln):
                pct_vals = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", next_ln)]
                pct_vals = [v for v in pct_vals if v > 0]
                if pct_vals:
                    discount_percent = pct_vals[0]

        # Some PDFs split amounts like "7 50.00" for "750.00" or "8 25.00" for "825.00".
        # Merge these *before* tokenization to avoid qty/rate/amount misalignment.
        ln = re.sub(r"\b(\d{1,3})\s+(\d{2}\.\d{2})\b", r"\1\2", ln)

        parts = [p for p in re.split(r"\s+", ln) if p]
        if len(parts) < 4:
            continue

        # Identify unit column position (PCS/NO) if present
        unit_idx = None
        for i, p in enumerate(parts):
            if p.lower() in unit_tokens:
                unit_idx = i
                break

        # Find numeric tokens
        numeric_idx = [i for i, p in enumerate(parts) if num.match(p)]
        if len(numeric_idx) < 3:
            continue

        # MAPPING RULES:
        # - If unit token (PCS/NO) exists: prefer <qty> <PCS/NO> <rate> <amount>
        # - Otherwise (table-like rows): infer qty/rate/amount from the LAST numeric tokens,
        #   preferring combinations where qty * rate ~= amount.
        if unit_idx is not None and unit_idx > 0:
            qty_i = unit_idx - 1 if num.match(parts[unit_idx - 1]) else numeric_idx[0]
            after = [j for j in numeric_idx if j > unit_idx]
            if len(after) >= 2:
                rate_i, amount_i = after[0], after[-1]
            else:
                # fallback to first/second/third numbers
                qty_i, rate_i, amount_i = numeric_idx[0], numeric_idx[1], numeric_idx[2]
        else:
            # Amount is usually the last numeric token (often with commas)
            amount_i = numeric_idx[-1]
            # Consider up to the last 5 numeric tokens before amount (excluding the amount itself)
            cand_idx = [j for j in numeric_idx[:-1]][-5:]
            best = None  # (score, penalty, qty_i, rate_i)
            amount_val = None
            try:
                amount_val = float(parts[amount_i].replace(",", ""))
            except Exception:
                amount_val = None

            # If we can read amount, try to find (qty, rate) that matches amount
            if amount_val is not None and cand_idx:
                for qi in cand_idx:
                    for ri in cand_idx:
                        if ri == qi:
                            continue
                        try:
                            qv = float(parts[qi].replace(",", ""))
                            rv = float(parts[ri].replace(",", ""))
                        except Exception:
                            continue
                        if qv <= 0 or rv <= 0:
                            continue
                        score = abs((qv * rv) - amount_val)
                        # Tie-breaker: prefer "reasonable" quantities (smaller integers)
                        penalty = 0.0
                        # quantities are usually smaller than rates in these bills
                        if qv > rv:
                            penalty += 1.0
                        if qv > 1000:
                            penalty += 2.0
                        if abs(qv - round(qv)) < 0.01:
                            penalty -= 0.25  # prefer integer qty
                        if best is None or (score, penalty) < (best[0], best[1]):
                            best = (score, penalty, qi, ri)

            if best is not None and best[0] <= max(1.0, (amount_val or 0) * 0.02):
                qty_i, rate_i = best[2], best[3]
            else:
                # Default to last three numbers: <qty> <rate> <amount>
                if len(numeric_idx) >= 3:
                    qty_i, rate_i = numeric_idx[-3], numeric_idx[-2]
                else:
                    qty_i, rate_i = numeric_idx[0], numeric_idx[1]

            amount_i = numeric_idx[-1]

        qty_tok = parts[qty_i].replace(",", "")
        rate_tok = parts[rate_i].replace(",", "")
        amount_tok = parts[amount_i].replace(",", "")

        try:
            qty = float(qty_tok)
        except Exception:
            continue
        try:
            rate = float(rate_tok)
        except Exception:
            rate = 0.0
        try:
            amount = float(amount_tok)
        except Exception:
            amount = 0.0

        # Item name is everything before the first numeric column among qty/rate
        name_end = min(qty_i, rate_i) if isinstance(rate_i, int) else qty_i
        name = " ".join(parts[:name_end]).strip()
        # Drop leading serial number from name if present (e.g. "1 Trophys TK 4085 D")
        name_parts = name.split()
        if len(name_parts) >= 2 and re.fullmatch(r"\d{1,3}", name_parts[0]):
            name = " ".join(name_parts[1:]).strip()
        if not name or len(name) < 2:
            continue
        if name.lower() in skip_exact or "total" in name.lower():
            continue

        # Validation rule: no '%' should remain inside item_name
        if "%" in name:
            name = name.replace("%", "").strip()

        items.append(
            LineItem(
                item_name=name,
                quantity=qty,
                rate=rate,
                discount_percent=discount_percent,
                amount=amount,
            )
        )
        if len(items) >= max_items:
            break

    return items


def _fallback_extract_gst_flattened_items(text: str, max_items: int = 50) -> List[LineItem]:
    """
    STRICT Type 4 parsing:
      '1 3926 No 1 700.00 700.00 9% 63.00'
    Map:
      qty = number AFTER unit (No)
      rate = next number
      amount = next number
    Never treat HSN (3926) as qty.
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: List[LineItem] = []

    for ln in lines:
        ln2 = re.sub(r"\s+", " ", ln).strip()
        parts = [p for p in ln2.replace(",", "").split(" ") if p]
        if len(parts) < 7:
            continue

        # Find unit token (No/Nos)
        unit_idx = None
        for i, p in enumerate(parts):
            if p.lower() in ("no", "nos", "no."):
                unit_idx = i
                break
        if unit_idx is None or unit_idx + 3 >= len(parts):
            continue

        # HSN is typically the token just before unit
        hsn_idx = unit_idx - 1
        if hsn_idx <= 0 or not re.fullmatch(r"\d{3,8}", parts[hsn_idx]):
            continue

        # qty is the number AFTER unit (critical for GST-flattened)
        if not re.fullmatch(r"\d+(?:\.\d+)?", parts[unit_idx + 1]):
            continue
        qty = float(parts[unit_idx + 1])

        # Next numeric tokens after qty: rate, amount (skip duplicates like "700 700.00")
        nums_after = []
        for tok in parts[unit_idx + 2 :]:
            if re.fullmatch(r"\d+(?:\.\d+)?", tok):
                nums_after.append(float(tok))
        if len(nums_after) < 2:
            continue
        rate = nums_after[0]
        amount = nums_after[1]

        # Item name: tokens between serial number and HSN
        item_name = " ".join(parts[1:hsn_idx]).strip() or "Item"
        if "total" in item_name.lower():
            continue

        out.append(LineItem(item_name=item_name, quantity=qty, rate=rate, amount=amount))
        if len(out) >= max_items:
            break

    return out


def _extract_row_discounts_from_text(text: str, max_rows: int = 100) -> List[float]:
    """
    Extract discount percentages from small-font percent-only lines that follow a row.

    Many invoices render discount in a second line like:
      1 ITEM ... 2,808
      (10%) (0%)

    We take the first positive percentage from that follow-up line.
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: List[float] = []
    percent_line_re = re.compile(r"^\s*(?:\(?\s*\d+(?:\.\d+)?\s*%\s*\)?\s*)+$")
    skip_prefix = ("total", "grand total", "received amount", "previous balance", "current balance")

    for i, ln in enumerate(lines):
        low = ln.lower().strip()
        if any(low.startswith(p) for p in skip_prefix):
            continue
        if "invoice amount in words" in low:
            continue

        is_rowish = (
            bool(re.search(r"\b(?:pcs?|nos?|no\.?)\b", low))
            or len(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", ln)) >= 3
        )
        if not is_rowish:
            continue

        discount = 0.0
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if percent_line_re.match(nxt):
                pct_vals = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", nxt)]
                pct_vals = [v for v in pct_vals if 0 < v <= 100]
                if pct_vals:
                    discount = pct_vals[0]

        out.append(discount)
        if len(out) >= max_rows:
            break

    return out


def _apply_text_row_adjustments(items: List[LineItem], text: str) -> None:
    """
    Use the original cleaned text to correct row-level discounts and net amounts.
    """
    discounts = _extract_row_discounts_from_text(text, max_rows=len(items))
    if not discounts:
        return

    for item, discount in zip(items, discounts):
        if discount <= 0:
            continue

        gross = float(item.quantity or 0) * float(item.rate or 0)
        if gross <= 0:
            if item.discount_percent <= 0:
                item.discount_percent = discount
            continue

        expected_net = gross * (1 - discount / 100)
        discount_value = gross * discount / 100

        if item.discount_percent <= 0:
            item.discount_percent = discount

        # If the extracted amount is actually the discount value, replace it with net amount.
        if item.amount > 0 and abs(item.amount - discount_value) <= max(1.0, gross * 0.02):
            item.amount = expected_net
        elif item.amount <= 0:
            item.amount = expected_net


def _fallback_extract_additional_charges(text: str, max_items: int = 20) -> List[AdditionalCharge]:
    """
    Best-effort extraction for non-product rows such as freight, packing and P&F.

    This is intentionally conservative:
    - only lines containing charge keywords are considered
    - product-like rows are ignored
    - the amount is taken from the last money-like value on the line
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: List[AdditionalCharge] = []
    charge_kw = (
        "freight",
        "fright",
        "courier",
        "p&f",
        "p & f",
        "packing",
        "forwarding",
        "transport",
        "transportation",
        "shipping",
        "handling",
        "charges",
        "charge",
        "round off",
        "roundoff",
    )
    money_re = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?")
    qty_unit_re = re.compile(r"\b(no|nos|no\.|pcs|pc)\b", flags=re.I)

    for ln in lines:
        low = ln.lower()
        if not any(k in low for k in charge_kw):
            continue
        if low.startswith(("total", "grand total", "received amount")):
            continue

        nums = money_re.findall(ln.replace(",", ""))
        if not nums:
            continue

        # Name: trim serial numbers and trailing numeric clutter.
        name = ln
        name = re.sub(r"^\s*\d{1,3}\s+", "", name)
        if qty_unit_re.search(name):
            name = re.split(r"\b(?:no\.?|nos?|pcs|pc)\b", name, maxsplit=1, flags=re.I)[0].strip()
        name = re.split(r"\s+[-:]\s*₹?\s*\d", name, maxsplit=1)[0].strip()
        name = re.sub(r"\s+", " ", name).strip()

        if not name:
            continue

        # Prefer the last numeric value on the row.
        try:
            amount = float(nums[-1].replace(",", ""))
        except Exception:
            continue
        if amount <= 0:
            continue

        # If the line is of the form "P & F - 150 - ₹ 150", keep the visible charge label only.
        out.append(AdditionalCharge(charge_name=name, amount=amount))
        if len(out) >= max_items:
            break

    return out


class AIExtractor:
    """
    AI-powered data extractor for financial documents.
    
    Uses Groq AI (Llama 3) for intelligent extraction.
    Extracts structured data from invoices, bills, and purchase documents.
    
    This class is designed to be stateless - no data is cached or stored.
    """
    
    def __init__(self):
        """Initialize the AI extractor with Groq client."""
        # Pin model via GROQ_MODEL env var; default is a stable general extractor.
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("[AI_EXTRACTOR] GROQ_API_KEY not found in environment. Please set it in .env file.")
        
        self.groq_client = Groq(api_key=api_key)
        print(f"[AI_EXTRACTOR] Groq AI initialized with model: {self.model}")
    
    def extract(self, text_content: str, tables: list = None) -> ExtractedData:
        """
        Extract structured data from document text using AI.
        
        Args:
            text_content: Raw text extracted from document
            tables: Optional list of tables (DataFrames) from document (not used, kept for compatibility)
            
        Returns:
            ExtractedData object with extracted fields
        """
        if not text_content:
            result = ExtractedData()
            result.extraction_notes.append("No content provided for extraction")
            return result
        
        # Adaptive preprocessing (format-aware, pattern-based)
        header = _extract_header_fields(text_content or "")
        cleaned_text, detected_format = adaptive_preprocess(text_content or "", max_chars=4000)
        table_block = _render_tables_for_prompt(tables)

        logger.info(
            "extract_start model=%s text_len=%s cleaned_len=%s tables=%s approx_tokens=%s",
            self.model,
            len(text_content or ""),
            len(cleaned_text or ""),
            len(tables or []),
            _approx_tokens(cleaned_text) + _approx_tokens(table_block),
        )
        logger.info("detected_format=%s", detected_format)
        logger.info("cleaned_preview=%r", (cleaned_text or "")[:240])

        _safe_print(f"\n{'='*60}")
        _safe_print(f"[AI_EXTRACTOR] Starting AI extraction")
        _safe_print(f"[AI_EXTRACTOR] Cleaned text length: {len(cleaned_text)} chars")
        _safe_print(f"[AI_EXTRACTOR] API Key loaded? {'Yes' if self.groq_client.api_key else 'NO'}")
        _safe_print(f"{'='*60}")
        _safe_print(f"[AI_EXTRACTOR] CLEANED TEXT PREVIEW: {cleaned_text[:200]}...")
        
        try:
            # Type 4 (GST flattened): do NOT rely fully on LLM; prefer fallback first.
            if detected_format == "gst_flattened":
                fb = _fallback_extract_gst_flattened_items(cleaned_text) or _fallback_extract_line_items(cleaned_text)
                charges = _fallback_extract_additional_charges(cleaned_text)
                result = ExtractedData(
                    invoice_number=header.get("invoice_number", ""),
                    date=header.get("date", ""),
                    vendor_name=header.get("vendor_name", ""),
                    customer_name=header.get("customer_name", ""),
                    cgst=float(header.get("cgst", 0) or 0),
                    sgst=float(header.get("sgst", 0) or 0),
                    igst=float(header.get("igst", 0) or 0),
                    total=float(header.get("total", 0) or 0),
                    extraction_notes=[f"detected_format={detected_format}", "Used fallback-first extraction for GST-flattened rows."]
                )
                result.tax = result.cgst + result.sgst + result.igst
                if fb:
                    result.line_items = fb
                if charges:
                    result.additional_charges = charges
                _apply_text_row_adjustments(result.line_items, cleaned_text)
                return result

            prompt = f"""You are a Forensic Document Analyzer. Your job is to perform a DEEP SCAN of this document and extract data with 100% FATAL PRECISION.

STRUCTURED TABLES (if present; highest priority for line-items):
{table_block}

DOCUMENT TEXT:
{cleaned_text}

Return a JSON object with this exact structure:
{{
    "invoice_number": "string",
    "date": "string",
    "vendor_name": "string",
    "customer_name": "string",
    "line_items": [
        {{
            "item_name": "string",
            "quantity": number,
            "rate": number,
            "discount_percent": number,
            "amount": number
        }}
    ],
    "additional_charges": [
        {{
            "charge_name": "string",
            "quantity": number,
            "rate": number,
            "amount": number
        }}
    ],
    "subtotal": number,
    "cgst": number,
    "sgst": number,
    "igst": number,
    "total": number
}}

*** DEEP ANALYSIS PROTOCOLS (STRICT ADHERENCE REQUIRED) ***

1. **VERBATIM EXTRACTION (NO SUMMARIZATION)**
   - **Vendor Name**: The company **issuing** the invoice (seller/supplier). Full legal name from header/footer.
   - **Customer Name**: For sales/tax invoices, the **buyer** / **Bill To** / **Consignee (Ship to)** / **Details of Receiver** — whoever is being charged. Leave "" on purchase bills if only the supplier is shown.
   - **Invoice Number**: Capture every character, symbol, and digit. (e.g., "GST/2024-25/001" -> extract fully).
   - **Item Names**: Product/SKU lines from the item table ONLY. Include model numbers, codes, sizes, brands. Do NOT put buyer names, school names, delivery addresses, or city lines in `item_name` — those belong in `customer_name` only. Never duplicate a standalone discount percentage (e.g. `50` or `50%`) inside `item_name`; put it in `discount_percent` only.

2. **INTELLIGENT COLUMN MAPPING**
   - The text might be jumbled. Look for patterns:
     - "Item | HSN | Qty | Rate | Amount" -> identifying the structure is key.
     - Do NOT confuse "HSN Code" (usually 4-8 digits) with "Rate" or "Quantity".
     - Do NOT confuse "Serial Number" (1, 2, 3) with "Quantity".

3. **DATE PRECISION**
   - Look for "Invoice Date", "Bill Date", or "Date".
   - If multiple dates exist (e.g., "PO Date", "Challan Date"), IGNORE THEM. Only extract the **Invoice/Bill Date**.

4. **FINANCIAL MATH & LOGIC**
   - **Tax Rule**: If there is no explicit Tax Amount column with non-zero values, Tax is 0. Do not calculate it yourself based on percentages found in small print.
   - **Total Verification**: The "Grand Total" printed on the paper is the Truth. Your calculated total MUST match it.

5. **DISCOUNTS & ADJUSTMENTS**
   - Search deeply for "Less:", "Discount", "Round Off".
   - "Round Off" should be treated as an additional charge (can be negative).
   - IMPORTANT: In many PDFs, discounts appear as a standalone line like `(50%)` immediately after an item row.
     In that case, assign **discount_percent=50** to the PREVIOUS item row. Do NOT create a new line item for `(50%)`.

6. **NO HALLUCINATIONS**
   - If a field is missing, return empty string "" or 0. Do NOT guess.

Perform this deep analysis now. Return ONLY valid JSON."""

            # Call Groq with retry/backoff for 429 and transient errors.
            _safe_print(f"[AI_EXTRACTOR] Calling Groq AI...")
            last_exc: Optional[Exception] = None
            response_text = ""

            for attempt in range(1, 4):
                try:
                    response = self.groq_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert Indian invoice extractor.\n"
                                    "CRITICAL RULES:\n"
                                    "1) Always return VALID JSON.\n"
                                    "2) If there is evidence of item rows, extract at least 1 best-guess line item.\n"
                                    "3) Do NOT mix charges (freight, P&F, GST, courier) with products.\n"
                                    "4) vendor_name = issuing company (seller); customer_name = buyer/bill-to/consignee.\n"
                                    "5) Standalone discount lines like (50%) belong to the previous item row.\n"
                                    "6) Do NOT confuse GST % with discount %.\n"
                                    "7) All numeric fields must be literal numbers (e.g. 2231.6). NEVER output math like `24194 - 21962.4`.\n"
                                )
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=4096,
                    )
                    response_text = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    last_exc = e
                    msg = str(e).lower()
                    is_429 = ("429" in msg) or ("rate limit" in msg) or ("ratelimit" in msg)
                    if is_429 and attempt < 3:
                        backoff = 2 ** attempt  # 2s, 4s, 8s
                        logger.warning("groq_rate_limit attempt=%s backoff_s=%s err=%r", attempt, backoff, str(e)[:200])
                        time.sleep(backoff)
                        continue
                    logger.exception("groq_call_failed attempt=%s", attempt)
                    raise

            if not response_text and last_exc:
                raise last_exc
            
            _safe_print(f"[AI_EXTRACTOR] Groq response received ({len(response_text)} chars)")
            _safe_print(f"[AI_EXTRACTOR] RAW RESPONSE: {response_text}")  # Log raw response to stdout
            logger.info("raw_llm_response_len=%s", len(response_text))



            # Parse JSON response (robust)
            data = _safe_json_loads(response_text)
            if data is None:
                result = ExtractedData()
                result.error_code = "llm_parse_error"
                result.raw_llm_response = response_text[:8000]
                result.extraction_notes.append(f"Extracted using Groq AI (model={self.model})")
                result.extraction_notes.append(f"detected_format={detected_format}")
                result.extraction_notes.append("llm_parse_error: Could not parse JSON response")
                # Header fallback
                result.invoice_number = header.get("invoice_number", "")
                result.date = header.get("date", "")
                result.vendor_name = header.get("vendor_name", "")
                result.customer_name = header.get("customer_name", "")
                try:
                    result.total = float(header.get("total", 0) or 0)
                except Exception:
                    pass
                # Fallback extraction from cleaned text
                result.line_items = _fallback_extract_line_items(cleaned_text)
                result.additional_charges = _fallback_extract_additional_charges(cleaned_text)
                _apply_text_row_adjustments(result.line_items, cleaned_text)
                if result.line_items:
                    result.extraction_notes.append("Fallback extraction succeeded (regex-based).")
                return result
            
            # Convert to ExtractedData with pricing and GST
            cgst = float(data.get("cgst", 0) or 0)
            sgst = float(data.get("sgst", 0) or 0)
            igst = float(data.get("igst", 0) or 0)
            # Total tax is sum of GST components
            total_tax = cgst + sgst + igst
            
            result = ExtractedData(
                invoice_number=data.get("invoice_number", ""),
                date=data.get("date", ""),
                vendor_name=data.get("vendor_name", ""),
                customer_name=data.get("customer_name", "") or "",
                subtotal=float(data.get("subtotal", 0) or 0),
                cgst=cgst,
                sgst=sgst,
                igst=igst,
                tax=total_tax,
                total=float(data.get("total", 0) or 0),
                extraction_notes=[f"Extracted using Groq AI (model={self.model})", f"detected_format={detected_format}"]
            )
            result.raw_llm_response = response_text[:8000]
            # Fill missing header fields from regex extraction
            if not result.invoice_number:
                result.invoice_number = header.get("invoice_number", "")
            if not result.date:
                result.date = header.get("date", "")
            if not result.vendor_name:
                result.vendor_name = header.get("vendor_name", "")
            if not result.customer_name:
                result.customer_name = header.get("customer_name", "")
            if not result.total and header.get("total"):
                try:
                    result.total = float(header.get("total", 0) or 0)
                except Exception:
                    pass
            
            # Keywords that indicate a charge (not a product)
            CHARGE_KEYWORDS = [
                'packing', 'forwarding', 'freight', 'fright', 'shipping', 'handling',
                'delivery', 'transport', 'transportation', 'courier',
                'service charge', 'service fee', 'insurance', 'loading',
                'unloading', 'charges', 'charge', 'p&f', 'p & f',
                'round off', 'roundoff', 'discount', 'less :', 'less:'
            ]
            
            def is_charge(item_name: str) -> bool:
                """Check if an item name looks like a charge/fee rather than a product."""
                name_lower = item_name.lower()
                return any(keyword in name_lower for keyword in CHARGE_KEYWORDS)
            
            # Parse line items with pricing and discount percentage
            for item in data.get("line_items", []):
                qty = float(item.get("quantity", 1) or 1)
                rate = float(item.get("rate", 0) or 0)
                discount_percent = float(item.get("discount_percent", 0) or 0)
                amount = float(item.get("amount", 0) or 0)
                item_name = item.get("item_name", "Unknown")
                item_name, discount_percent = _split_trailing_discount_from_item_name(
                    item_name, discount_percent
                )

                # Fix common swapped/misread columns:
                # Only infer quantity from amount/rate when there is no discount.
                # Discounted rows often have amount = qty * rate * (1 - discount),
                # so using amount/rate would incorrectly halve the true quantity.
                if amount > 0 and rate > 0 and discount_percent <= 0:
                    inferred_qty = amount / rate
                    if (
                        qty <= 20
                        and inferred_qty > 0
                        and abs(inferred_qty - round(inferred_qty)) < 0.01
                        and abs((round(inferred_qty) * rate) - amount) <= max(1.0, amount * 0.01)
                    ):
                        qty = float(int(round(inferred_qty)))
                # If rate is likely wrong (e.g. very small) but amount/qty is clean, infer rate.
                if amount > 0 and qty > 0 and (rate <= 0 or rate < 1):
                    inferred_rate = amount / qty
                    if inferred_rate > 0:
                        rate = float(inferred_rate)
                
                # If amount is 0 but we have qty and rate, calculate it with percentage discount
                if amount == 0 and rate > 0:
                    if discount_percent > 0:
                        amount = qty * rate * (1 - discount_percent / 100)
                    else:
                        amount = qty * rate
                
                # PHANTOM DISCOUNT CHECK:
                # If parsed amount roughly equals (qty * rate), then NO discount was applied.
                # If AI extracted a discount % (like 18%) but the math shows no discount, it's false positive (likely GST).
                if amount > 0 and rate > 0 and discount_percent > 0:
                    expected = qty * rate
                    # Allow small rounding difference (e.g. 1.0)
                    if abs(expected - amount) < 1.0:
                        _safe_print(f"   [DISCOUNT CORRECTION] Removed false {discount_percent}% discount for '{item_name}' (Math proves no discount)")
                        discount_percent = 0.0
                
                # Filter out summary/total rows masquerading as items
                nm_low = (item_name or "").lower().strip()
                if nm_low.startswith(("total", "grand total", "received amount", "previous balance", "current balance")):
                    continue

                # Post-processing: Check if this should be a charge instead of a line item
                if is_charge(item_name):
                    # Move to additional_charges instead
                    result.additional_charges.append(AdditionalCharge(
                        charge_name=item_name,
                        amount=amount
                    ))
                    _safe_print(f"   [CHARGE DETECTED] '{item_name}' moved to additional_charges")
                else:
                    result.line_items.append(LineItem(
                        item_name=item_name,
                        quantity=qty,
                        rate=rate,
                        discount_percent=discount_percent,
                        amount=amount
                    ))

            # Text-level correction pass for layouts that use a tiny follow-up discount line.
            # This fixes cases where the model reads the discount value as the amount.
            _apply_text_row_adjustments(result.line_items, cleaned_text)
            
            # Parse additional_charges from AI response
            for charge in data.get("additional_charges", []):
                charge_name = charge.get("charge_name", "")
                charge_amount = float(charge.get("amount", 0) or 0)
                charge_qty = float(charge.get("quantity", 0) or 0)
                charge_rate = float(charge.get("rate", 0) or 0)
                
                if charge_name and charge_amount > 0:
                    result.additional_charges.append(AdditionalCharge(
                        charge_name=charge_name,
                        amount=charge_amount,
                        quantity=charge_qty,
                        rate=charge_rate
                    ))

            # Fallback charge extraction: catches rows the model skipped, especially freight/P&F lines.
            fallback_charges = _fallback_extract_additional_charges(cleaned_text)
            if fallback_charges:
                seen = {
                    (c.charge_name.lower().strip(), round(float(c.amount or 0), 2))
                    for c in result.additional_charges
                }
                for charge in fallback_charges:
                    key = (charge.charge_name.lower().strip(), round(float(charge.amount or 0), 2))
                    if key in seen:
                        continue
                    result.additional_charges.append(charge)
                    seen.add(key)

            _safe_print(f"[AI_EXTRACTOR] OK Extraction successful. Found {len(result.line_items)} items, {len(result.additional_charges)} charges")
            
            # Debug: Print extracted items with prices and discount percentage
            for i, item in enumerate(result.line_items):
                _safe_print(f"   {i+1}. {item.item_name} | Qty: {item.quantity} | Rate: {item.rate} | Disc: {item.discount_percent}% | Amount: {item.amount}")
            
            # Debug: Print charges
            if result.additional_charges:
                _safe_print(f"   Additional Charges:")
                for charge in result.additional_charges:
                    _safe_print(f"      - {charge.charge_name}: {charge.amount}")
            
            if result.total > 0:
                _safe_print(f"   Document Total: {result.total}")

            # Monitoring note: compare extracted items vs rough expected row count (non-fatal).
            expected_rows = _estimate_line_item_rows(cleaned_text)
            if expected_rows > 0 and len(result.line_items) < max(1, expected_rows // 3):
                result.extraction_notes.append(
                    f"LOW_CONFIDENCE: Extracted {len(result.line_items)} items but document looks like ~{expected_rows} rows. Check table parsing / layout."
                )
            if table_block:
                result.extraction_notes.append("Used structured PDF tables to guide extraction.")

            # If model returned empty line_items but we likely have rows, fallback.
            if not result.line_items and expected_rows > 0:
                fb = _fallback_extract_line_items(cleaned_text)
                if fb:
                    result.line_items = fb
                    result.extraction_notes.append("Fallback extraction used (LLM returned empty line_items).")
                else:
                    result.error_code = "no_line_items"
            
            return result
            
        except json.JSONDecodeError as e:
            _safe_print(f"[AI_EXTRACTOR] Failed to parse AI response as JSON: {e}")
            result = ExtractedData()
            result.extraction_notes.append(f"JSON parsing error: {e}")
            result.error_code = "llm_parse_error"
            return result
        except Exception as e:
            _safe_print(f"[AI_EXTRACTOR] AI extraction failed: {e}")
            result = ExtractedData()
            result.extraction_notes.append(f"Extraction error: {e}")
            msg = str(e).lower()
            if ("429" in msg) or ("rate limit" in msg) or ("ratelimit" in msg):
                result.error_code = "rate_limit_error"
                # Mandatory safety net: still try fallback extraction so pipeline never empties out.
                try:
                    cleaned_text, detected_format = adaptive_preprocess(text_content or "", max_chars=4000)
                    if detected_format == "gst_flattened":
                        fb = _fallback_extract_gst_flattened_items(cleaned_text) or _fallback_extract_line_items(cleaned_text)
                    else:
                        fb = _fallback_extract_line_items(cleaned_text)
                    if fb:
                        result.line_items = fb
                        result.extraction_notes.append("Used fallback extraction due to rate limit.")
                        result.extraction_notes.append(f"detected_format={detected_format}")
                    result.additional_charges = _fallback_extract_additional_charges(cleaned_text)
                    _apply_text_row_adjustments(result.line_items, cleaned_text)
                except Exception:
                    pass
            else:
                result.error_code = "extraction_error"
            return result
