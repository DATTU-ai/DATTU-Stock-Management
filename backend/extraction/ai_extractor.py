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
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Load environment variables from backend/.env regardless of launch directory
from dotenv import load_dotenv

_BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_BACKEND_ENV_PATH)

# Groq AI client
from groq import Groq


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
        
        _safe_print(f"\n{'='*60}")
        _safe_print(f"[AI_EXTRACTOR] Starting AI extraction")
        _safe_print(f"[AI_EXTRACTOR] Text length: {len(text_content)} chars")
        _safe_print(f"[AI_EXTRACTOR] API Key loaded? {'Yes' if self.groq_client.api_key else 'NO'}")  # Check if key exists
        _safe_print(f"{'='*60}")
        
        # Log preview of text to stdout (Render logs)
        _safe_print(f"[AI_EXTRACTOR] TEXT PREVIEW: {text_content[:200]}...")
        
        try:
            # Prefer structured tables when available (reduces line-wrap / discount-on-next-line failures).
            table_block = _render_tables_for_prompt(tables)
            prompt = f"""You are a Forensic Document Analyzer. Your job is to perform a DEEP SCAN of this document and extract data with 100% FATAL PRECISION.

STRUCTURED TABLES (if present; highest priority for line-items):
{table_block}

DOCUMENT TEXT:
{text_content}

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

            _safe_print(f"[AI_EXTRACTOR] Calling Groq AI...")
            
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Indian invoice extractor. Extract GST invoices. CRITICAL: (1) vendor_name = issuing company (seller); customer_name = buyer/bill-to/consignee when present. (2) line item item_name must be products only—never buyer school names or addresses. (3) Do NOT confuse GST % with line discount %. (4) Trailing discount like 50%% must be discount_percent, not part of item_name."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4096
            )
            
            response_text = response.choices[0].message.content.strip()
            _safe_print(f"[AI_EXTRACTOR] Groq response received ({len(response_text)} chars)")
            _safe_print(f"[AI_EXTRACTOR] RAW RESPONSE: {response_text}")  # Log raw response to stdout



            # Clean up response - remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            # Parse JSON response
            data = json.loads(response_text)
            
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
                extraction_notes=[f"Extracted using Groq AI (model={self.model})"]
            )
            
            # Keywords that indicate a charge (not a product)
            CHARGE_KEYWORDS = [
                'packing', 'forwarding', 'freight', 'shipping', 'handling',
                'delivery', 'transport', 'transportation', 'courier',
                'service charge', 'service fee', 'insurance', 'loading',
                'unloading', 'charges', 'charge', 'p&f', 'p & f'
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
            expected_rows = _estimate_line_item_rows(text_content)
            if expected_rows > 0 and len(result.line_items) < max(1, expected_rows // 3):
                result.extraction_notes.append(
                    f"LOW_CONFIDENCE: Extracted {len(result.line_items)} items but document looks like ~{expected_rows} rows. Check table parsing / layout."
                )
            if table_block:
                result.extraction_notes.append("Used structured PDF tables to guide extraction.")
            
            return result
            
        except json.JSONDecodeError as e:
            _safe_print(f"[AI_EXTRACTOR] Failed to parse AI response as JSON: {e}")
            result = ExtractedData()
            result.extraction_notes.append(f"JSON parsing error: {e}")
            return result
        except Exception as e:
            _safe_print(f"[AI_EXTRACTOR] AI extraction failed: {e}")
            result = ExtractedData()
            result.extraction_notes.append(f"Extraction error: {e}")
            return result

