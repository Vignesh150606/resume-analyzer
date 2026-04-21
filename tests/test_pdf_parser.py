import sys
import json
sys.path.insert(0, ".")

from backend.core.pdf_parser import extract_text_from_pdf, extract_sections, PDFParsingError

def test_extraction():
    print("=" * 60)
    print("TEST 1: Basic extraction")
    print("=" * 60)
    
    try:
        result = extract_text_from_pdf("sample_data/sample_resume.pdf")
        
        print(f"Pages found     : {result['page_count']}")
        print(f"Characters found: {result['char_count']}")
        print(f"\nFirst 500 characters of extracted text:")
        print("-" * 40)
        print(result['full_text'][:500])
        print("-" * 40)
        
    except PDFParsingError as e:
        print(f"PDFParsingError: {e}")
        return

    print("\n")
    print("=" * 60)
    print("TEST 2: Section detection")
    print("=" * 60)
    
    sections = extract_sections(result['full_text'])
    print(f"Sections detected: {list(sections.keys())}")
    
    for section_name, content in sections.items():
        print(f"\n[{section_name.upper()}]")
        print(content[:200])

    print("\n")
    print("=" * 60)
    print("TEST 3: Error handling")
    print("=" * 60)
    
    try:
        extract_text_from_pdf("nonexistent.pdf")
    except PDFParsingError as e:
        print(f"Correctly caught missing file: {e}")
    
    try:
        extract_text_from_pdf("requirements.txt")
    except PDFParsingError as e:
        print(f"Correctly caught wrong file type: {e}")

    print("\nAll tests passed.")

if __name__ == "__main__":
    test_extraction()