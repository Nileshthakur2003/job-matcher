import io
import logging
from typing import Optional
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts plain text from raw PDF file bytes using PyMuPDF (fitz).
    Preserves document structure and cleans excessive whitespaces.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        extracted_pages = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")
            if text and text.strip():
                extracted_pages.append(text.strip())

        doc.close()
        full_text = "\n\n".join(extracted_pages)
        return full_text.strip()
    except Exception as e:
        logger.error(f"Failed to parse PDF with PyMuPDF: {e}")
        # Return fallback empty or basic string
        return ""
