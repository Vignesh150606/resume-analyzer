from fastapi import FastAPI
from backend.core.config import get_settings

app = FastAPI(
    title="Resume Analyzer API",
    description="AI-powered resume analysis backend",
    version="0.1.0"
)

settings = get_settings()


@app.get("/")
def health_check():
    return {
        "status": "running",
        "app": settings.app_name,
        "environment": settings.app_env
    }


@app.get("/health")
def health():
    return {"healthy": True}