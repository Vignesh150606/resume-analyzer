import os
import uuid
import logging
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import get_settings
from backend.core.pipeline import run_full_analysis
from backend.core.pdf_parser import PDFParsingError
from backend.models.schemas import FullAnalysisResult

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="Resume Analyzer API",
    description="AI-powered resume analysis using NLP and LLM",
    version="1.0.0"
)

# CORS — allows the Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "running",
        "app": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"healthy": True}


@app.post("/analyze", response_model=FullAnalysisResult)
async def analyze_resume(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text")
):
    """
    Main analysis endpoint.

    Accepts:
        - resume: PDF file (multipart upload)
        - job_description: Plain text of the job description

    Returns:
        Complete FullAnalysisResult with scores, skill gaps, and LLM feedback
    """
    # Validate file type
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file."
        )

    # Validate JD length
    if len(job_description.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Job description is too short. Please provide at least 50 characters."
        )

    if len(job_description) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Job description is too long. Maximum 10,000 characters."
        )

    # Save uploaded PDF to a temp file
    # We can't pass the file stream directly to pdfplumber — it needs a path
    tmp_path = None
    try:
        suffix = f"_{uuid.uuid4().hex}.pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await resume.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Processing resume: {resume.filename} ({len(content)} bytes)")

        # Run the full pipeline
        result = run_full_analysis(
            pdf_path=tmp_path,
            jd_text=job_description
        )

        return result

    except PDFParsingError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )

    finally:
        # Always clean up the temp file — even if analysis failed
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info("Temp file cleaned up")