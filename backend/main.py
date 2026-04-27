"""
Document-to-Excel Processing API
================================
FastAPI backend for converting bill and purchase documents into structured Excel files.

Key Features:
- Stateless, synchronous API
- Privacy-first: ZERO data storage
- All processing in-memory
- Streaming Excel response
- Sales/Purchase bill analysis with surplus/deficit

Endpoints:
- POST /analyze-bills - Analyze multiple sales/purchase bills

Author: Antigravity AI Platform
"""

import io
import gc
import asyncio
import json as _json
import random
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

# Import processing modules
from parsers import DocumentParser
from extraction import AIExtractor
from validation import Validator
from generators import ExcelGenerator
from analysis import InventoryAnalyzer

# Import authentication routes
from routes.auth import router as auth_router


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Document-to-Excel Processor",
    description="Privacy-first API for converting invoices and bills to structured Excel files",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
# Explicitly list allowed origins to avoid browser blocking
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173", 
    "https://dattu-stock-management-qww1.onrender.com",  # Your production frontend
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routes
app.include_router(auth_router)

# Initialize processing components (stateless)
document_parser = DocumentParser()
ai_extractor = AIExtractor()
validator = Validator()
excel_generator = ExcelGenerator()
inventory_analyzer = InventoryAnalyzer()

# Serialize LLM calls across requests (production stability under load)
LLM_LOCK = asyncio.Lock()
logger = logging.getLogger("invoice_processing")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    Health check endpoint.
    
    Returns basic API information.
    """
    return {
        "status": "healthy",
        "service": "Document-to-Excel Processor",
        "version": "1.0.0",
        "privacy": "All data is processed in-memory and never stored"
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check for monitoring.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "parser": "ready",
            "extractor": "ready",
            "validator": "ready",
            "generator": "ready"
        }
    }





# ============================================================================
# Multi-Bill Analysis Endpoint
# ============================================================================

@app.post("/analyze-bills")
async def analyze_bills(
    purchase_files: List[UploadFile] = File(default=[]),
    sales_files: List[UploadFile] = File(default=[]),
    auto_detect: bool = Form(default=True)
):
    """
    Analyze multiple purchase and sales bills.
    
    Accepts:
    - Multiple purchase bill files
    - Multiple sales bill files
    - Auto-detect option for bill type classification
    
    Returns:
    - Excel file with:
      - Inventory Summary (surplus/deficit per item)
      - Purchase Bills details
      - Sales Bills details
      - AI Analysis & Insights
    
    Privacy:
    - All processing in-memory
    - No data stored or logged
    """
    
    purchase_data: List[dict] = []
    sales_data: List[dict] = []
    failed_bills: List[dict] = []
    skipped_rate_limit: List[dict] = []
    total_bills = len(purchase_files) + len(sales_files)
    
    try:
        # Process purchase files
        for file in purchase_files:
            try:
                file_bytes = await file.read()
                if len(file_bytes) == 0:
                    continue
                    
                parse_result = document_parser.parse(file_bytes, file.filename)
                
                # DEBUG: Print what was extracted from PDF
                print(f"Processing Purchase File: {file.filename}")
                
                if parse_result.success:
                    # Rate limit control + sequential LLM calls across requests
                    async with LLM_LOCK:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                        extracted = await run_in_threadpool(
                            ai_extractor.extract,
                            parse_result.text_content,
                            parse_result.tables,
                        )

                    err_code = getattr(extracted, "error_code", "") or ""
                    if err_code == "rate_limit_error":
                        # If we still extracted items via fallback, treat as processed (do not count as skipped)
                        if extracted.line_items:
                            failed_bills.append({
                                "kind": "PURCHASE",
                                "filename": file.filename,
                                "invoice_number": extracted.invoice_number,
                                "date": extracted.date,
                                "reason": "rate_limit_error (used fallback extraction)"
                            })
                        else:
                            skipped_rate_limit.append({
                                "kind": "PURCHASE",
                                "filename": file.filename,
                                "invoice_number": extracted.invoice_number,
                                "date": extracted.date,
                                "reason": "skipped_due_to_rate_limit"
                            })
                    elif err_code:
                        failed_bills.append({
                            "kind": "PURCHASE",
                            "filename": file.filename,
                            "invoice_number": extracted.invoice_number,
                            "date": extracted.date,
                            "reason": f"{err_code}: " + ("; ".join(extracted.extraction_notes) if extracted.extraction_notes else "")
                        })
                    elif not extracted.line_items:
                        failed_bills.append({
                            "kind": "PURCHASE",
                            "filename": file.filename,
                            "invoice_number": extracted.invoice_number,
                            "date": extracted.date,
                            "reason": "no_line_items: " + ("; ".join(extracted.extraction_notes) if extracted.extraction_notes else "")
                        })

                    # Production behavior: include bill even if line_items is empty (for output visibility).
                    purchase_data.append({
                        'invoice_number': extracted.invoice_number,
                        'date': extracted.date,
                        'vendor_name': extracted.vendor_name,
                        'customer_name': extracted.customer_name,
                        'line_items': extracted.line_items,
                        'additional_charges': extracted.additional_charges,
                        'subtotal': extracted.subtotal,
                        'cgst': extracted.cgst,
                        'sgst': extracted.sgst,
                        'igst': extracted.igst,
                        'tax': extracted.tax,
                        'total': extracted.total
                    })
                else:
                    # Soft failure: record and continue (never fail the entire request)
                    failed_bills.append({
                        "kind": "PURCHASE",
                        "filename": file.filename,
                        "invoice_number": "",
                        "date": "",
                        "reason": f"file_parse_error: {parse_result.error_message}",
                    })
                    continue
            except Exception as e:
                failed_bills.append({
                    "kind": "PURCHASE",
                    "filename": file.filename,
                    "invoice_number": "",
                    "date": "",
                    "reason": f"file_parse_error: {str(e)}",
                })
                continue
        
        # Process sales files
        for file in sales_files:
            try:
                file_bytes = await file.read()
                if len(file_bytes) == 0:
                    continue
                    
                parse_result = document_parser.parse(file_bytes, file.filename)
                
                print(f"Processing Sales File: {file.filename}")
                
                if parse_result.success:
                    async with LLM_LOCK:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                        extracted = await run_in_threadpool(
                            ai_extractor.extract,
                            parse_result.text_content,
                            parse_result.tables,
                        )

                    err_code = getattr(extracted, "error_code", "") or ""
                    if err_code == "rate_limit_error":
                        if extracted.line_items:
                            failed_bills.append({
                                "kind": "SALES",
                                "filename": file.filename,
                                "invoice_number": extracted.invoice_number,
                                "date": extracted.date,
                                "reason": "rate_limit_error (used fallback extraction)"
                            })
                        else:
                            skipped_rate_limit.append({
                                "kind": "SALES",
                                "filename": file.filename,
                                "invoice_number": extracted.invoice_number,
                                "date": extracted.date,
                                "reason": "skipped_due_to_rate_limit"
                            })
                    elif err_code:
                        failed_bills.append({
                            "kind": "SALES",
                            "filename": file.filename,
                            "invoice_number": extracted.invoice_number,
                            "date": extracted.date,
                            "reason": f"{err_code}: " + ("; ".join(extracted.extraction_notes) if extracted.extraction_notes else "")
                        })
                    elif not extracted.line_items:
                        failed_bills.append({
                            "kind": "SALES",
                            "filename": file.filename,
                            "invoice_number": extracted.invoice_number,
                            "date": extracted.date,
                            "reason": "no_line_items: " + ("; ".join(extracted.extraction_notes) if extracted.extraction_notes else "")
                        })
                    
                    # IMPORTANT:
                    # The UI already separates Purchase vs Sales uploads.
                    # Respect that split and do not re-route sales uploads into purchases.
                    sales_data.append({
                        'invoice_number': extracted.invoice_number,
                        'date': extracted.date,
                        'vendor_name': extracted.vendor_name,
                        'customer_name': extracted.customer_name,
                        'line_items': extracted.line_items,
                        'additional_charges': extracted.additional_charges,
                        'subtotal': extracted.subtotal,
                        'cgst': extracted.cgst,
                        'sgst': extracted.sgst,
                        'igst': extracted.igst,
                        'tax': extracted.tax,
                        'total': extracted.total
                    })
                else:
                    failed_bills.append({
                        "kind": "SALES",
                        "filename": file.filename,
                        "invoice_number": "",
                        "date": "",
                        "reason": f"file_parse_error: {parse_result.error_message}",
                    })
                    continue
            except Exception as e:
                failed_bills.append({
                    "kind": "SALES",
                    "filename": file.filename,
                    "invoice_number": "",
                    "date": "",
                    "reason": f"file_parse_error: {str(e)}",
                })
                continue
        
        # Partial success support: only fail if nothing was uploaded.
        if total_bills == 0:
            raise HTTPException(status_code=400, detail="No bills uploaded.")
        
        # Perform inventory analysis
        analysis = inventory_analyzer.analyze(purchase_data, sales_data)
        
        bills_all = purchase_data + sales_data
        successful = len([b for b in bills_all if b.get("line_items")])
        partial = len([b for b in bills_all if (not b.get("line_items")) and (b.get("invoice_number") or b.get("vendor_name") or b.get("total"))])
        failed = len(failed_bills)
        skipped = len(skipped_rate_limit)
        summary = {
            "total_bills": total_bills,
            "successful": successful,
            "partial": partial,
            "failed": failed,
            "skipped_due_to_rate_limit": skipped,
        }

        # Generate Excel report
        excel_bytes = excel_generator.generate_analysis_report(
            analysis,
            purchase_data,
            sales_data,
            failed_bills=(failed_bills + skipped_rate_limit),
            run_summary=summary,
        )
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"inventory_analysis_{timestamp}.xlsx"
        
        # Build response headers
        response_headers = {
            "Content-Disposition": f'attachment; filename="{output_filename}"',
            "Content-Length": str(len(excel_bytes)),
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Processing-Summary": _json.dumps(summary),
        }
        
        # Return streaming response
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=response_headers
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        import traceback
        print(f"ERROR in analyze-bills: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing bills: {str(e)}"
        )
        
    finally:
        gc.collect()


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom error handler for HTTP exceptions.
    
    Returns clean JSON errors without exposing internals.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Catch-all error handler.
    
    Never exposes internal error details for privacy/security.
    """
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred. Please try again.",
            "status_code": 500
        }
    )


# ============================================================================
# Run Configuration
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    # In production, use: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True  # Disable in production
    )
