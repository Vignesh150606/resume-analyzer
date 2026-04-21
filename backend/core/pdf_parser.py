import pdfplumber
import re
import logging
from pathlib import Path

# Set up logging — professional code always logs, never just prints
logger = logging.getLogger(__name__)


class PDFParsingError(Exception):
    """
    Custom exception for PDF parsing failures.
    Always create custom exceptions for your modules —
    it makes error handling cleaner and more specific.
    """
    pass


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract and clean text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with:
            - full_text: Complete cleaned text
            - pages: List of text per page
            - page_count: Total number of pages
            - char_count: Total character count
            
    Raises:
        PDFParsingError: If file doesn't exist, isn't a PDF,
                        or contains no extractable text
    """
    path = Path(pdf_path)
    
    # Validate file exists
    if not path.exists():
        raise PDFParsingError(f"File not found: {pdf_path}")
    
    # Validate file extension
    if path.suffix.lower() != ".pdf":
        raise PDFParsingError(f"File is not a PDF: {pdf_path}")
    
    # Validate file size (reject empty files and files > 10MB)
    file_size = path.stat().st_size
    if file_size == 0:
        raise PDFParsingError("PDF file is empty")
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise PDFParsingError("PDF file too large (max 10MB)")
    
    pages_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                raise PDFParsingError("PDF has no pages")
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    # extract_text() returns None if page has no text
                    raw_text = page.extract_text()
                    
                    if raw_text:
                        cleaned = clean_text(raw_text)
                        pages_text.append(cleaned)
                    else:
                        # Log which pages had no text — useful for debugging
                        logger.warning(
                            f"Page {page_num + 1} has no extractable text. "
                            f"It may be a scanned image."
                        )
                        pages_text.append("")
                        
                except Exception as e:
                    logger.error(f"Failed to extract page {page_num + 1}: {e}")
                    pages_text.append("")
                    
    except PDFParsingError:
        raise  # Re-raise our custom errors without wrapping them
    except Exception as e:
        raise PDFParsingError(f"Failed to open PDF: {str(e)}")
    
    # Combine all pages
    full_text = "\n\n".join(page for page in pages_text if page.strip())
    
    # Reject if we got nothing useful
    if not full_text.strip():
        raise PDFParsingError(
            "No text could be extracted. "
            "This PDF may be scanned (image-based). "
            "Please upload a digital PDF."
        )
    
    logger.info(
        f"Successfully extracted {len(full_text)} characters "
        f"from {len(pdf.pages)} pages"
    )
    
    return {
        "full_text": full_text,
        "pages": pages_text,
        "page_count": len(pages_text),
        "char_count": len(full_text)
    }


def clean_text(text: str) -> str:
    """
    Clean raw extracted text from pdfplumber.
    
    This function handles the common messiness of PDF extraction:
    - Multiple spaces between words
    - Multiple blank lines
    - Weird unicode characters
    - Windows-style line endings
    """
    if not text:
        return ""
    
    # Normalize line endings (Windows PDFs use \r\n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Replace non-breaking spaces and other unicode spaces with regular space
    text = text.replace("\xa0", " ").replace("\u200b", "")
    
    # Remove multiple spaces but preserve single spaces
    text = re.sub(r" {2,}", " ", text)
    
    # Remove more than 2 consecutive newlines (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Clean up lines — strip trailing/leading whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    return text.strip()


def extract_sections(full_text: str) -> dict:
    """
    Attempt to identify common resume sections by looking for
    section headers. This is heuristic-based — it won't be perfect
    for all resume formats, but covers the majority.
    
    Returns a dict of section_name -> section_content
    """
    # Common resume section headers (case insensitive)
    section_patterns = [
        r"(education|academic)",
        r"(experience|work experience|employment)",
        r"(skills|technical skills|core competencies)",
        r"(projects|personal projects|academic projects)",
        r"(certifications?|courses?|training)",
        r"(summary|objective|profile|about)",
        r"(achievements?|awards?|honors?)",
        r"(languages?)",
        r"(contact|personal information)",
    ]
    
    sections = {}
    lines = full_text.split("\n")
    current_section = "header"
    current_content = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if this line is a section header
        matched_section = None
        for pattern in section_patterns:
            if re.search(pattern, line_lower):
                # Only treat as header if line is short (headers are short)
                if len(line.split()) <= 5:
                    matched_section = line.strip()
                    break
        
        if matched_section:
            # Save the previous section
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = matched_section.lower()
            current_content = []
        else:
            current_content.append(line)
    
    # Don't forget the last section
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()
    
    return sections