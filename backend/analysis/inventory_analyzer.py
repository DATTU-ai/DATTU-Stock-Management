"""
Inventory Analyzer Module
=========================
Analyzes Sales and Purchase bills to calculate inventory metrics.

Features:
- Aggregates items from multiple bills
- Calculates total purchased vs sold quantities
- Determines surplus/deficit per item
- Generates AI-powered insights and recommendations

All processing is done in-memory. No data is stored.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class BillType(Enum):
    """Type of bill - Sales or Purchase."""
    SALES = "sales"
    PURCHASE = "purchase"
    UNKNOWN = "unknown"


class StockStatus(Enum):
    """Stock status based on surplus/deficit."""
    SURPLUS = "surplus"      # More purchased than sold
    DEFICIT = "deficit"      # More sold than purchased
    BALANCED = "balanced"    # Equal purchase and sale
    LOW_STOCK = "low_stock"  # Surplus but below threshold


@dataclass
class InventoryItem:
    """
    Aggregated inventory data for a single item.
    
    Attributes:
        item_name: Normalized lowercase key used for matching across bills
        display_label: Best human-readable label (first seen raw name, preserves e.g. ATF)
        purchased_qty: Total quantity purchased
        sold_qty: Total quantity sold
        surplus_deficit: purchased_qty - sold_qty
        status: Stock status (surplus/deficit/balanced)
        purchased_value: Total value of purchases
        sold_value: Total value of sales
    """
    item_name: str
    display_label: str = ""
    purchased_qty: float = 0.0
    sold_qty: float = 0.0
    surplus_deficit: float = 0.0
    status: StockStatus = StockStatus.BALANCED
    purchased_value: float = 0.0
    sold_value: float = 0.0


@dataclass
class InventoryAnalysis:
    """
    Complete inventory analysis result.
    
    Attributes:
        items: List of all items with inventory data
        surplus_items: Items with surplus stock
        deficit_items: Items with deficit stock
        low_stock_items: Items with low stock (< threshold)
        top_selling_items: Top selling items by quantity
        insights: Generated insights and recommendations
        purchase_bill_count: Number of purchase bills
        sales_bill_count: Number of sales bills
        total_purchase_value: Total value of all purchases
        total_sales_value: Total value of all sales
        purchase_date_range: Tuple of (start_date, end_date) for purchase bills
        sales_date_range: Tuple of (start_date, end_date) for sales bills
        date_mismatch_warning: Warning message if date ranges don't match
    """
    items: List[InventoryItem] = field(default_factory=list)
    surplus_items: List[str] = field(default_factory=list)
    deficit_items: List[str] = field(default_factory=list)
    low_stock_items: List[str] = field(default_factory=list)
    top_selling_items: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    purchase_bill_count: int = 0
    sales_bill_count: int = 0
    total_purchase_value: float = 0.0
    total_sales_value: float = 0.0
    # Date range tracking
    purchase_date_range: tuple = field(default_factory=lambda: (None, None))
    sales_date_range: tuple = field(default_factory=lambda: (None, None))
    date_mismatch_warning: str = ""


class InventoryAnalyzer:
    """
    Analyzes inventory from sales and purchase bills.
    
    Stateless processor that aggregates item data from multiple
    bills and calculates surplus/deficit metrics.
    """
    
    # Threshold for low stock warning
    LOW_STOCK_THRESHOLD = 10
    
    # Keywords to detect bill type
    SALES_KEYWORDS = [
        'sold to', 'customer', 'invoice to', 'bill to', 
        'ship to', 'buyer', 'sales invoice', 'tax invoice',
        'retail', 'sale'
    ]
    
    PURCHASE_KEYWORDS = [
        'purchased from', 'supplier', 'vendor', 'purchase order',
        'po number', 'bought from', 'purchase invoice',
        'wholesale', 'procurement'
    ]
    
    def detect_bill_type(self, text_content: str) -> BillType:
        """
        Auto-detect bill type from content.
        
        Args:
            text_content: Raw text from document
            
        Returns:
            BillType enum (SALES, PURCHASE, or UNKNOWN)
        """
        text_lower = text_content.lower()
        
        sales_score = sum(1 for kw in self.SALES_KEYWORDS if kw in text_lower)
        purchase_score = sum(1 for kw in self.PURCHASE_KEYWORDS if kw in text_lower)
        
        if sales_score > purchase_score:
            return BillType.SALES
        elif purchase_score > sales_score:
            return BillType.PURCHASE
        else:
            return BillType.UNKNOWN
    
    def analyze(
        self,
        purchase_data: List[Dict],
        sales_data: List[Dict]
    ) -> InventoryAnalysis:
        """
        Analyze inventory from purchase and sales data.
        
        Args:
            purchase_data: List of extracted data from purchase bills
                Each dict has: line_items, subtotal, tax, total, etc.
            sales_data: List of extracted data from sales bills
            
        Returns:
            InventoryAnalysis with stock metrics
        """
        result = InventoryAnalysis()
        result.purchase_bill_count = len(purchase_data)
        result.sales_bill_count = len(sales_data)
        
        # Extract and validate date ranges
        purchase_dates = self._extract_dates(purchase_data)
        sales_dates = self._extract_dates(sales_data)
        
        # Set date ranges
        if purchase_dates:
            result.purchase_date_range = (min(purchase_dates), max(purchase_dates))
        if sales_dates:
            result.sales_date_range = (min(sales_dates), max(sales_dates))
        
        # Validate date range match
        result.date_mismatch_warning = self._validate_date_ranges(
            result.purchase_date_range, 
            result.sales_date_range
        )
        
        # Aggregate items by name
        item_map: Dict[str, InventoryItem] = {}
        
        # Process purchase bills - quantities and values
        for bill in purchase_data:
            line_items = bill.get('line_items', [])
            
            for item in line_items:
                name = self._normalize_item_name(item.item_name)
                if name not in item_map:
                    raw = (getattr(item, "item_name", None) or "").strip()
                    item_map[name] = InventoryItem(
                        item_name=name,
                        display_label=raw or name,
                    )
                elif not item_map[name].display_label:
                    raw = (getattr(item, "item_name", None) or "").strip()
                    if raw:
                        item_map[name].display_label = raw
                
                item_map[name].purchased_qty += item.quantity
                # Add item value if available
                if hasattr(item, 'amount') and item.amount > 0:
                    item_map[name].purchased_value += item.amount
                    result.total_purchase_value += item.amount
        
        # Process sales bills - quantities and values
        for bill in sales_data:
            line_items = bill.get('line_items', [])
            
            for item in line_items:
                name = self._normalize_item_name(item.item_name)
                if name not in item_map:
                    raw = (getattr(item, "item_name", None) or "").strip()
                    item_map[name] = InventoryItem(
                        item_name=name,
                        display_label=raw or name,
                    )
                elif not item_map[name].display_label:
                    raw = (getattr(item, "item_name", None) or "").strip()
                    if raw:
                        item_map[name].display_label = raw
                
                item_map[name].sold_qty += item.quantity
                # Add item value if available
                if hasattr(item, 'amount') and item.amount > 0:
                    item_map[name].sold_value += item.amount
                    result.total_sales_value += item.amount
        
        # Calculate metrics for each item
        for name, item in item_map.items():
            # Calculate surplus/deficit
            item.surplus_deficit = item.purchased_qty - item.sold_qty
            
            # Determine status
            if item.surplus_deficit > 0:
                if item.surplus_deficit < self.LOW_STOCK_THRESHOLD:
                    item.status = StockStatus.LOW_STOCK
                    result.low_stock_items.append(item.display_label or name)
                else:
                    item.status = StockStatus.SURPLUS
                    result.surplus_items.append(item.display_label or name)
            elif item.surplus_deficit < 0:
                item.status = StockStatus.DEFICIT
                result.deficit_items.append(item.display_label or name)
            else:
                item.status = StockStatus.BALANCED
        
        # Sort by sold quantity for top sellers
        sorted_items = sorted(
            item_map.values(),
            key=lambda x: x.sold_qty,
            reverse=True
        )
        # Filter for items with significant sales (> 50)
        # User defined criteria: "items which have saled with more than 50 quantity"
        top_sellers = [item for item in sorted_items if item.sold_qty > 50]
        
        result.top_selling_items = [
            (item.display_label or item.item_name) for item in top_sellers[:10]
        ]
        result.items = list(item_map.values())
        
        # Generate insights
        result.insights = self._generate_insights(result)
        
        return result
    
    def _extract_dates(self, bill_data: List[Dict]) -> List[str]:
        """
        Extract all dates from bill data.
        
        Args:
            bill_data: List of bill dictionaries
            
        Returns:
            List of date strings found
        """
        dates = []
        for bill in bill_data:
            date_str = bill.get('date', '')
            if date_str and date_str.strip():
                # Normalize date format
                normalized = self._normalize_date(date_str)
                if normalized:
                    dates.append(normalized)
        return dates
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """
        Normalize date string to a consistent format.
        
        Handles various Indian date formats:
        - DD/MM/YYYY
        - DD-MM-YYYY  
        - YYYY-MM-DD
        - DD Mon YYYY
        
        Returns:
            Normalized date string (DD/MM/YYYY) or None if parsing fails
        """
        import re
        from datetime import datetime
        
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # Try various formats
        formats = [
            ("%d/%m/%Y", r'\d{1,2}/\d{1,2}/\d{4}'),
            ("%d-%m-%Y", r'\d{1,2}-\d{1,2}-\d{4}'),
            ("%Y-%m-%d", r'\d{4}-\d{1,2}-\d{1,2}'),
            ("%d %b %Y", r'\d{1,2}\s+\w{3}\s+\d{4}'),
            ("%d %B %Y", r'\d{1,2}\s+\w+\s+\d{4}'),
            # Support for "8-Apr-25" type formats
            ("%d-%b-%y", r'\d{1,2}-\w{3}-\d{2}'),
            ("%d-%b-%Y", r'\d{1,2}-\w{3}-\d{4}'),
        ]
        
        for fmt, pattern in formats:
            try:
                # Try to parse
                dt = datetime.strptime(date_str, fmt)
                # Return in consistent format DD/MM/YYYY
                return dt.strftime("%d/%m/%Y")
            except ValueError:
                continue
        
        # If all formats fail, return original if it looks like a date
        if re.match(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', date_str):
            return date_str
        
        return None
    
    def _validate_date_ranges(
        self, 
        purchase_range: tuple, 
        sales_range: tuple
    ) -> str:
        """
        Validate that purchase and sales date ranges match.
        
        Args:
            purchase_range: (start_date, end_date) tuple for purchases
            sales_range: (start_date, end_date) tuple for sales
            
        Returns:
            Warning message if ranges don't match, empty string otherwise
        """
        p_start, p_end = purchase_range
        s_start, s_end = sales_range
        
        # If either is missing, can't validate
        if not p_start or not s_start:
            return ""
        
        # Check if ranges match
        if p_start != s_start or p_end != s_end:
            warning = (
                f"DATE MISMATCH WARNING: Purchase bills are from {p_start} to {p_end}, "
                f"but Sales bills are from {s_start} to {s_end}. "
                f"For accurate inventory analysis, ensure both bill sets cover the same date range."
            )
            return warning
        
        return ""
    
    def _normalize_item_name(self, name: str) -> str:
        """
        Normalize item name for matching.
        
        Converts to lowercase, removes extra whitespace, strips size tokens,
        and standardizes common code variants so purchase/sales match better.
        """
        if not name:
            return "Unknown Item"
        import re

        s = ' '.join(name.lower().strip().split())

        # Remove common size tokens like 9.5", 11.5", 12", 9.5 inch
        s = re.sub(r"\b\d+(?:\.\d+)?\s*(?:\"|inches|inch|in)\b", "", s)
        s = re.sub(r"\b\d+(?:\.\d+)?\s*'\s*\b", "", s)
        s = re.sub(r"\s+", " ", s).strip()

        # Standardize "at <digits> <letter>" -> "atf <digits> <letter>"
        # (helps merge AT vs ATF codes across different invoice templates)
        tokens = s.split()
        if len(tokens) >= 3 and tokens[0] == "at" and tokens[1].isdigit() and len(tokens[2]) <= 2:
            tokens[0] = "atf"
            s = " ".join(tokens)

        # Remove stray punctuation around codes
        s = re.sub(r"[^\w\s/.-]+", "", s).strip()
        s = re.sub(r"\s+", " ", s).strip()

        return s
    
    def _generate_insights(self, analysis: InventoryAnalysis) -> List[str]:
        """
        Generate AI-powered insights based on analysis.
        
        Uses rule-based logic to provide recommendations.
        Uses Excel-compatible symbols for proper display.
        """
        insights = []
        
        # Deficit warnings - Critical alert
        if analysis.deficit_items:
            insights.append(
                f"[CRITICAL] Stock Deficit: {len(analysis.deficit_items)} items have been "
                f"sold more than purchased. Immediate action needed! See Deficit Items below."
            )
        
        # Low stock warnings
        if analysis.low_stock_items:
            insights.append(
                f"[ALERT] Low Stock: {len(analysis.low_stock_items)} items have less than "
                f"{self.LOW_STOCK_THRESHOLD} units remaining. Consider reordering soon!"
            )
        
        # Surplus items - Positive indicator
        if analysis.surplus_items:
            insights.append(
                f"[GOOD] Surplus Stock: {len(analysis.surplus_items)} items have healthy excess inventory. "
                f"Good stock levels maintained."
            )
        
        # Top sellers
        if analysis.top_selling_items:
            insights.append(
                f"[TOP] Bestsellers: {len(analysis.top_selling_items)} top performing items identified. "
                f"See Top Selling Items below."
            )
        
        # Summary stats
        insights.append(
            f"[SUMMARY] Analyzed {analysis.purchase_bill_count} purchase bills + "
            f"{analysis.sales_bill_count} sales bills = {len(analysis.items)} unique items tracked."
        )
        
        return insights

