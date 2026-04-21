from pydantic import BaseModel, Field
from typing import Optional


class SkillExtractionResult(BaseModel):
    all_skills: list[str]
    by_category: dict[str, list[str]]
    skill_count: int


class SkillComparisonResult(BaseModel):
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    match_percentage: float
    matched_by_category: dict[str, list[str]]
    missing_by_category: dict[str, list[str]]


class SectionScore(BaseModel):
    section_name: str
    similarity_score: float


class FinalScore(BaseModel):
    final_score: float
    grade: str
    label: str
    breakdown: dict[str, float]


class ImprovementSuggestion(BaseModel):
    section: str = Field(description="Which resume section this applies to")
    issue: str = Field(description="What the specific problem is")
    suggestion: str = Field(description="Exactly what to add or change")
    priority: str = Field(description="high / medium / low")


class LLMAnalysis(BaseModel):
    """
    Structured output from Gemini.
    Every field is typed and validated — no surprises.
    """
    overall_assessment: str = Field(
        description="2-3 sentence honest assessment of resume fit"
    )
    top_strengths: list[str] = Field(
        description="3 specific strengths relevant to this JD",
        min_length=1,
        max_length=5
    )
    critical_gaps: list[str] = Field(
        description="Top 3 most important missing skills or experiences",
        min_length=1,
        max_length=5
    )
    improvement_suggestions: list[ImprovementSuggestion] = Field(
        description="Specific, actionable improvements",
        min_length=1,
        max_length=5
    )
    ats_keywords_to_add: list[str] = Field(
        description="Keywords from JD missing in resume that ATS would scan for"
    )
    rewritten_summary: str = Field(
        description="A rewritten professional summary optimized for this JD"
    )
    hiring_probability: str = Field(
        description="low / medium / high — likelihood of getting shortlisted"
    )


class FullAnalysisResult(BaseModel):
    """
    The complete analysis result returned by the entire pipeline.
    This is what the API returns to the frontend.
    """
    # Input metadata
    resume_char_count: int
    resume_page_count: int

    # Skill analysis
    resume_skills: SkillExtractionResult
    jd_skills: SkillExtractionResult
    skill_comparison: SkillComparisonResult

    # Similarity scores
    semantic_score: float
    section_scores: dict[str, float]
    final_score: FinalScore

    # LLM analysis
    llm_analysis: LLMAnalysis

    # Skill gap details
    skill_gap_details: list[dict]