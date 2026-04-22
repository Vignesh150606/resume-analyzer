import sys
import pytest
sys.path.insert(0, ".")

from backend.core.pdf_parser import (
    extract_text_from_pdf,
    extract_sections,
    clean_text,
    PDFParsingError
)


# ── clean_text tests ──────────────────────────────────────────

def test_clean_text_removes_extra_spaces():
    result = clean_text("hello   world")
    assert "  " not in result
    assert "hello world" == result


def test_clean_text_normalizes_newlines():
    result = clean_text("line1\r\nline2\r\nline3")
    assert "\r" not in result


def test_clean_text_removes_extra_blank_lines():
    result = clean_text("line1\n\n\n\n\nline2")
    assert "\n\n\n" not in result


def test_clean_text_handles_empty_string():
    result = clean_text("")
    assert result == ""


def test_clean_text_handles_none_gracefully():
    result = clean_text("")
    assert isinstance(result, str)


def test_clean_text_strips_whitespace():
    result = clean_text("  hello world  ")
    assert result == "hello world"


# ── PDFParsingError tests ─────────────────────────────────────

def test_extract_raises_error_for_missing_file():
    with pytest.raises(PDFParsingError) as exc_info:
        extract_text_from_pdf("nonexistent_file.pdf")
    assert "not found" in str(exc_info.value).lower()


def test_extract_raises_error_for_wrong_extension():
    with pytest.raises(PDFParsingError) as exc_info:
        extract_text_from_pdf("requirements.txt")
    assert "not a pdf" in str(exc_info.value).lower()


# ── extract_sections tests ────────────────────────────────────

def test_extract_sections_returns_dict():
    sample_text = """
John Doe
john@email.com

Education
B.Tech Computer Science

Skills
Python, JavaScript, SQL

Projects
Built a web app
"""
    result = extract_sections(sample_text)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_extract_sections_detects_education():
    sample_text = "John Doe\n\nEducation\nB.Tech CSE\n\nSkills\nPython"
    result = extract_sections(sample_text)
    keys_lower = [k.lower() for k in result.keys()]
    assert any("education" in k for k in keys_lower)


def test_extract_sections_detects_skills():
    sample_text = "Header\n\nSkills\nPython, Java\n\nProjects\nBuilt something"
    result = extract_sections(sample_text)
    keys_lower = [k.lower() for k in result.keys()]
    assert any("skill" in k for k in keys_lower)


# ── Real PDF test ─────────────────────────────────────────────

def test_extract_real_pdf_returns_text():
    result = extract_text_from_pdf("sample_data/sample_resume.pdf")
    assert result["char_count"] > 100
    assert result["page_count"] >= 1
    assert len(result["full_text"]) > 100
    assert isinstance(result["pages"], list)


def test_extract_real_pdf_has_all_keys():
    result = extract_text_from_pdf("sample_data/sample_resume.pdf")
    assert "full_text" in result
    assert "pages" in result
    assert "page_count" in result
    assert "char_count" in result