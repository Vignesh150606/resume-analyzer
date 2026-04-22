# AI-Powered Resume Analyzer

An intelligent resume analysis system that compares resumes against job descriptions using NLP and LLM techniques.

## Live Demo
> Coming soon — deploy link will be added here

## Features
- PDF resume parsing with text extraction and section detection
- Skill extraction using a 150+ skill taxonomy with regex word-boundary matching
- Semantic similarity scoring using `sentence-transformers` (all-MiniLM-L6-v2)
- AI-powered structured feedback via Google Gemini LLM
- Match score (0-100) with letter grade and breakdown
- Missing skill detection and ATS keyword suggestions
- Clean Streamlit web interface with REST API backend

## Tech Stack
| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.12 |
| NLP | sentence-transformers, scikit-learn |
| LLM | Google Gemini API |
| PDF Parsing | pdfplumber |
| Validation | Pydantic v2 |
| Frontend | Streamlit |
| Testing | pytest (37 tests, 100% pass rate) |

## Project Structure
