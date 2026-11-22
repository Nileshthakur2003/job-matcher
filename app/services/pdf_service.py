import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts plain text from raw PDF file bytes using PyMuPDF (fitz).
    Preserves structural line breaks and strips extraneous whitespace.
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
        return "\n\n".join(extracted_pages).strip()
    except Exception as e:
        logger.error(f"Failed to parse PDF with PyMuPDF: {e}")
        return ""
