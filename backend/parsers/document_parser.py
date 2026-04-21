"""
Document Parser Module
======================
Handles file type detection and content extraction from various document formats.
Supports: Excel (.xlsx, .xls), PDF (text-based), Images (for future OCR support).

All processing is done in-memory using BytesIO buffers.
No data is written to disk at any point.
"""

import io
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import pandas as pd
# import pdfplumber
# from PIL import Image


class FileType(Enum):
    """Supported file types for document processing."""
    EXCEL = "excel"
    PDF = "pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class ParseResult:
    """
    Result of document parsing.
    
    Attributes:
        success: Whether parsing was successful
        file_type: Detected file type
        text_content: Extracted text content (from PDF or OCR)
        tables: List of extracted tables as DataFrames
        error_message: Error description if parsing failed
    """
    success: bool
    file_type: FileType
    text_content: str = ""
    tables: list = field(default_factory=list)
    error_message: str = ""


class DocumentParser:
    """
    Stateless document parser for extracting content from uploaded files.
    
    Supports:
    - Excel files: Direct table extraction
    - Text-based PDFs: Text and table extraction
    - Images: Currently returns placeholder (OCR not configured)
    
    All operations are performed in-memory with no disk writes.
    """
    
    # File signature (magic bytes) mapping for type detection
    # These are the first few bytes that identify file formats
    FILE_SIGNATURES = {
        # ZIP-based formats (xlsx, docx, etc.)
        b'PK\x03\x04': FileType.EXCEL,
        # PDF signature
        b'%PDF': FileType.PDF,
        # JPEG signature
        b'\xff\xd8\xff': FileType.IMAGE,
        # PNG signature
        b'\x89PNG': FileType.IMAGE,
        # Old Excel format (.xls)
        b'\xd0\xcf\x11\xe0': FileType.EXCEL,
    }
    
    def detect_file_type(self, file_bytes: bytes) -> FileType:
        """
        Detect file type from magic bytes (file signature).
        
        Args:
            file_bytes: Raw bytes of the uploaded file
            
        Returns:
            FileType enum indicating the detected format
        """
        for signature, file_type in self.FILE_SIGNATURES.items():
            if file_bytes.startswith(signature):
                return file_type
        return FileType.UNKNOWN
    
    def parse(self, file_bytes: bytes, filename: str = "") -> ParseResult:
        """
        Parse uploaded document and extract content.
        
        Args:
            file_bytes: Raw bytes of the uploaded file
            filename: Original filename (used as fallback for type detection)
            
        Returns:
            ParseResult containing extracted text and tables
        """
        # Detect file type from bytes
        file_type = self.detect_file_type(file_bytes)
        
        # Fallback: Use filename extension if magic bytes detection fails
        if file_type == FileType.UNKNOWN and filename:
            ext = filename.lower().split('.')[-1]
            if ext in ('xlsx', 'xls'):
                file_type = FileType.EXCEL
            elif ext == 'pdf':
                file_type = FileType.PDF
            elif ext in ('jpg', 'jpeg', 'png'):
                file_type = FileType.IMAGE
        
        # Route to appropriate parser
        try:
            if file_type == FileType.EXCEL:
                return self._parse_excel(file_bytes)
            elif file_type == FileType.PDF:
                return self._parse_pdf(file_bytes)
            elif file_type == FileType.IMAGE:
                return self._parse_image(file_bytes)
            else:
                return ParseResult(
                    success=False,
                    file_type=FileType.UNKNOWN,
                    error_message="Unsupported file format. Please upload Excel, PDF, or image files."
                )
        except Exception as e:
            return ParseResult(
                success=False,
                file_type=file_type,
                error_message=f"Failed to parse document: {str(e)}"
            )
    
    def _parse_excel(self, file_bytes: bytes) -> ParseResult:
        """
        Parse Excel file and extract all sheets as tables.
        
        Uses pandas for reading directly from BytesIO buffer.
        Each sheet becomes a separate DataFrame in the tables list.
        """
        buffer = io.BytesIO(file_bytes)
        tables = []
        text_parts = []
        
        try:
            # Read all sheets from the Excel file
            excel_data = pd.read_excel(buffer, sheet_name=None, engine='openpyxl')
            
            for sheet_name, df in excel_data.items():
                # Skip empty sheets
                if df.empty:
                    continue
                    
                # Store the DataFrame
                tables.append({
                    'sheet_name': sheet_name,
                    'data': df
                })
                
                # Convert to text representation for extraction
                text_parts.append(f"=== Sheet: {sheet_name} ===")
                text_parts.append(df.to_string(index=False))
            
            return ParseResult(
                success=True,
                file_type=FileType.EXCEL,
                text_content="\n\n".join(text_parts),
                tables=tables
            )
        finally:
            # Clear buffer from memory
            buffer.close()
    
    def _parse_pdf(self, file_bytes: bytes) -> ParseResult:
        """
        Parse PDF and extract text using pdfplumber.
        (Restored by user request for better accuracy).
        """
        buffer = io.BytesIO(file_bytes)
        text_content = []
        tables = []

        try:
            import pdfplumber
            with pdfplumber.open(buffer) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)

                    # Extract tables (best-effort). This is critical for invoices where
                    # discounts/taxes break onto separate lines in extracted text.
                    try:
                        page_tables = page.extract_tables(
                            table_settings={
                                "vertical_strategy": "lines",
                                "horizontal_strategy": "lines",
                                "intersection_tolerance": 5,
                                "snap_tolerance": 3,
                                "join_tolerance": 3,
                                "edge_min_length": 3,
                                "min_words_vertical": 1,
                                "min_words_horizontal": 1,
                            }
                        )
                    except Exception:
                        page_tables = None

                    if page_tables:
                        for table in page_tables:
                            if not table:
                                continue
                            # Convert raw table rows to DataFrame (keep all rows; headers may be messy)
                            try:
                                df = pd.DataFrame(table)
                                tables.append({"page": page_idx, "data": df})
                            except Exception:
                                continue

            full_text = "\n".join(text_content).strip()
            print(f"DEBUG: Extracted text length: {len(full_text)}")
            print(f"[PDF_PARSER] Extracted {len(full_text)} chars from PDF") 
            
            # Scanned PDF Detection:
            # If a PDF has 0 or remarkably few characters, it is likely an image scan.
            if len(full_text) < 50:
                 print(f"DEBUG: Scanned PDF detected! Text length: {len(full_text)}")
                 return ParseResult(
                    success=False,
                    file_type=FileType.PDF,
                    error_message="This appears to be a Scanned/Image-based PDF. This system currently only supports digital/text-based PDFs. Please upload the original digital file."
                )
            
            return ParseResult(
                success=True,
                file_type=FileType.PDF,
                text_content=full_text,
                tables=tables
            )
            
        except Exception as e:
            return ParseResult(
                success=False,
                file_type=FileType.PDF,
                error_message=f"PDF parsing failed: {str(e)}"
            )
        finally:
            buffer.close()

    def _parse_image(self, file_bytes: bytes) -> ParseResult:
        """
        Image parsing is DISABLED.
        """
        return ParseResult(
            success=False,
            file_type=FileType.IMAGE,
            error_message="Image processing is disabled. Please upload a valid PDF or Excel file."
        )
