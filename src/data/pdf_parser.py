import os
from pathlib import Path
from typing import Tuple, Optional
from pypdf import PdfReader


def parse_pdf_report(pdf_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Reads a clinical PDF report and extracts its textual content.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        (text, error_message): Extracted text string if successful, or None and the error message.
    """
    if not os.path.exists(pdf_path):
        return None, f"File not found: {pdf_path}"
        
    try:
        reader = PdfReader(pdf_path)
        if len(reader.pages) == 0:
            return None, "Empty PDF file (0 pages)"
            
        full_text = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
                
        combined_text = "\n".join(full_text).strip()
        if not combined_text:
            return None, "PDF contains no extractable text (likely scanned raster image)"
            
        return combined_text, None
        
    except Exception as e:
        return None, f"PDF extraction error: {str(e)}"
