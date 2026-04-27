"""
Validation Engine Module
========================
Validates extracted data for stock management.

Validation Rules:
1. Mandatory fields must be present
2. Quantity values must be non-negative
3. Item names should not be empty

All validation is performed in-memory. No data is logged or stored.
"""

from dataclasses import dataclass, field
from typing import List
from extraction import ExtractedData


@dataclass
class ValidationResult:
    """
    Result of data validation.
    
    Attributes:
        is_valid: Whether all validation rules passed
        errors: List of error messages for failed rules
        warnings: List of warning messages for minor issues
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str):
        """Add an error and mark result as invalid."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning (does not affect validity)."""
        self.warnings.append(message)


class Validator:
    """
    Validator for extracted stock management data.
    
    Validates extracted data to ensure data quality and consistency.
    Focused on stock management - validates items and quantities.
    
    This class is stateless - no data is cached or stored.
    """
    
    def validate(self, data: ExtractedData) -> ValidationResult:
        """
        Validate extracted data.
        
        Args:
            data: ExtractedData object to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()
        
        # Run all validation checks
        self._validate_mandatory_fields(data, result)
        self._validate_line_items(data, result)
        
        return result
    
    def _validate_mandatory_fields(self, data: ExtractedData, result: ValidationResult):
        """
        Check that essential fields are present.
        """
        # Optional fields - warning if missing
        if not data.invoice_number:
            result.add_warning("Invoice/Bill number is missing")
        
        if not data.date:
            result.add_warning("Document date is missing")
        
        if not data.vendor_name:
            result.add_warning("Vendor/Supplier name could not be identified")
        
        if not data.line_items:
            # Production behavior: do not reject entire bill if items are missing.
            result.add_warning("No line items detected")
    
    def _validate_line_items(self, data: ExtractedData, result: ValidationResult):
        """
        Validate individual line items.
        
        - Item name should not be empty
        - Quantity should be non-negative
        - Rate and amount should be non-negative
        """
        items_without_prices = 0
        
        for i, item in enumerate(data.line_items, 1):
            # Check item name
            if not item.item_name or item.item_name.strip() == '':
                result.add_warning(f"Line item {i}: Item name is empty")
            
            # Check quantity
            if item.quantity < 0:
                result.add_error(f"Line item {i}: Quantity cannot be negative")
            
            # Check rate (warning only - may not always be present)
            if item.rate < 0:
                result.add_error(f"Line item {i}: Rate cannot be negative")
            elif item.rate == 0:
                items_without_prices += 1
            
            # Check amount (warning only)
            if item.amount < 0:
                result.add_error(f"Line item {i}: Amount cannot be negative")
        
        # Add a single warning if many items are missing prices
        if items_without_prices > 0 and items_without_prices == len(data.line_items):
            result.add_warning("No price data could be extracted from the document")

