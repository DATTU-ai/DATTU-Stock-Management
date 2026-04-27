"""
Preprocessing & lightweight format detection for invoice text.

Goals:
- Reduce token usage (remove noise, cap length)
- Make line-items more parseable (merge broken lines)
- Adapt behavior for different invoice styles without vendor hardcoding
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class InvoiceFormat(str):
    MULTI_LINE_PURCHASE = "multi_line_purchase"
    GST_PURCHASE = "gst_purchase"
    SALES_CLEAN = "sales_clean"
    GST_FLATTENED = "gst_flattened"
    GENERIC = "generic"


_WS_RE = re.compile(r"\s+")


def classify_format(text: str) -> str:
    """
    Strict, pattern-based classifier (per real invoice issues):
    - multi_line_purchase: multiple short lines + PCS scattered (Asia Trophy style)
    - gst_purchase: contains HSN/GST/CGST/SGST (ESS style)
    - sales_clean: contains 'Golden Moment Group' (clean sales invoices)
    - gst_flattened: rows like '1 3926 No 1 700.00 700.00 9% 63.00'
    - generic: fallback
    """
    if not text:
        return InvoiceFormat.GENERIC

    lines = [ln for ln in text.splitlines() if ln.strip()]
    lb = len(lines)
    unit_hits = len(re.findall(r"\b(PCS|PC|NOS?|NO\.)\b", text, flags=re.I))
    gst_hits = len(re.findall(r"\b(HSN|GST|CGST|SGST|IGST|TAX)\b", text, flags=re.I))

    # GST-flattened row patterns (sales GST tables often look like):
    #   "1 Trophies 3926 No 1 700.00 700 700.00 9% 63.00 ..."
    #   "1 3926 No 1 700.00 700.00 9% 63.00 ..."
    # Broad detection (some PDFs include extra duplicated numeric columns and weird spacing)
    if (
        re.search(r"\b\d{3,8}\s+(?:acs\s+)?no\.?\s+\d+(?:\.\d+)?\b", text, flags=re.I)
        and re.search(r"\b\d+\s*%", text, flags=re.I)
    ):
        return InvoiceFormat.GST_FLATTENED
    
    # Clean sales invoices are issued by Golden Moment Group (issuer header), but many purchase invoices
    # mention Golden Moment Group as the customer ("Bill To"). Require issuer-style header cues.
    lowtxt = text.lower()
    if "golden moment group" in lowtxt and any(
        k in lowtxt for k in ("reg. address", "office address", "pasaydan", "kirti cooper", "nigdi pradhikaran")
    ):
        return InvoiceFormat.SALES_CLEAN

    if lb >= 25 and unit_hits >= 3:
        return InvoiceFormat.MULTI_LINE_PURCHASE
    if gst_hits >= 3:
        return InvoiceFormat.GST_PURCHASE
    return InvoiceFormat.GENERIC


def _normalize_lines(text: str) -> list[str]:
    text = text.replace("\t", " ")
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    # drop consecutive duplicates
    out: list[str] = []
    last = None
    for ln in lines:
        if ln == last:
            continue
        out.append(ln)
        last = ln
    return out


def _remove_noise(lines: list[str]) -> list[str]:
    noise_kw = (
        "bank details",
        "ifsc",
        "account no",
        "upi",
        "terms and conditions",
        "authorised signatory",
        "this is a computer generated",
        "declaration",
        "amount chargeable",
        "amount (in words)",
        "e. & o.e",
        "qr code",
    )
    return [ln for ln in lines if not any(k in ln.lower() for k in noise_kw)]


def _slice_item_region(lines: list[str]) -> list[str]:
    if not lines:
        return lines

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

    return lines[start:end] if end > start else lines


_HAS_QTY_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(PCS|PC|NOS?|NO\.|KG|GM|GMS|LTR|L)\b", flags=re.I
)
_HAS_MONEY_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")


def _merge_broken_lines(lines: list[str]) -> list[str]:
    """
    Merge broken description lines into the next line when the current line
    doesn't look like a row (no qty/unit and no money-ish values).

    This is intentionally conservative.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""

        cur_is_rowish = bool(_HAS_QTY_UNIT_RE.search(cur)) or (len(_HAS_MONEY_RE.findall(cur)) >= 2)
        nxt_is_rowish = bool(_HAS_QTY_UNIT_RE.search(nxt)) or (len(_HAS_MONEY_RE.findall(nxt)) >= 2)

        # If current is likely just a wrapped name/descriptor, append into next
        if (not cur_is_rowish) and nxt and nxt_is_rowish:
            lines[i + 1] = f"{cur} {nxt}"
            i += 1
            continue

        out.append(cur)
        i += 1

    return out


_UNIT_TRIGGER = re.compile(r"\b(PCS|PC|NOS?|NO\.|NO)\b", flags=re.I)
_CHARGE_LINE = re.compile(r"\b(p\s*&\s*f|p\s*&\s*f\s*-|freight|courier|charges?)\b", flags=re.I)


def _merge_until_unit(lines: list[str]) -> list[str]:
    """
    STRICT multi-line merge (Asia Trophy):
    Merge lines UNTIL a line contains PCS/NO.

    Example:
      AT 1408 B
      B-9.5"
      1 PCS 320 160
      (50%)
    becomes one item line:
      AT 1408 B B-9.5" 1 PCS 320 160 (50%)

    Charge lines (P&F/Freight/Courier/Charges) are kept as separate lines.
    """
    out: list[str] = []
    buf: list[str] = []

    serial_line = re.compile(r"^\s*\d{1,3}\s+\S+")

    for ln in lines:
        if _CHARGE_LINE.search(ln):
            # flush any buffered item
            if buf:
                out.append(" ".join(buf).strip())
                buf = []
            out.append(ln)
            continue

        # If a new serial-number row starts while we are still buffering a previous item
        # (common in table extractions where description lines appear without the qty/unit line nearby),
        # flush to avoid appending description of item A into item B.
        if buf and serial_line.match(ln):
            out.append(" ".join(buf).strip())
            buf = []

        buf.append(ln)
        if _UNIT_TRIGGER.search(ln):
            out.append(" ".join(buf).strip())
            buf = []

    if buf:
        out.append(" ".join(buf).strip())
    return out


def preprocess_multi_line(text: str, max_chars: int = 4000) -> str:
    lines = _normalize_lines(text)
    lines = _slice_item_region(lines)
    lines = _remove_noise(lines)
    # Apply strict multi-line merge rule
    lines = _merge_until_unit(lines)
    cleaned = "\n".join(lines).strip()
    return cleaned[:max_chars] if len(cleaned) > max_chars else cleaned


def preprocess_gst(text: str, max_chars: int = 4000) -> str:
    lines = _normalize_lines(text)
    # GST invoices often have lots of headers; slice and remove noise
    lines = _slice_item_region(lines)
    lines = _remove_noise(lines)
    # Drop GST summary/tax rows aggressively; keep likely product rows
    drop_kw = ("output", "cgst", "sgst", "igst", "tax amount", "hsn/sac", "taxable value", "rate %", "total:", "declaration")
    filtered: list[str] = []
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in drop_kw):
            continue
        filtered.append(ln)
    lines = filtered
    cleaned = "\n".join(lines).strip()
    return cleaned[:max_chars] if len(cleaned) > max_chars else cleaned


def preprocess_generic(text: str, max_chars: int = 4000) -> str:
    lines = _normalize_lines(text)
    lines = _slice_item_region(lines)
    lines = _remove_noise(lines)
    cleaned = "\n".join(lines).strip()
    return cleaned[:max_chars] if len(cleaned) > max_chars else cleaned


def preprocess(text: str, max_chars: int = 4000) -> tuple[str, str]:
    fmt = classify_format(text)
    if fmt == InvoiceFormat.MULTI_LINE_PURCHASE:
        return preprocess_multi_line(text, max_chars=max_chars), fmt
    if fmt == InvoiceFormat.GST_PURCHASE or fmt == InvoiceFormat.GST_FLATTENED:
        return preprocess_gst(text, max_chars=max_chars), fmt
    if fmt == InvoiceFormat.SALES_CLEAN:
        return preprocess_generic(text, max_chars=max_chars), fmt
    return preprocess_generic(text, max_chars=max_chars), fmt

