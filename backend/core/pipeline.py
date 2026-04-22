import logging
from backend.core.pdf_parser import extract_text_from_pdf, extract_sections
from backend.core.extractor import extract_skills, compare_skills, extract_experience_years
from backend.core.similarity import (
    compute_overall_semantic_score,
    compute_section_similarities,
    compute_skill_gap_embeddings,
    compute_final_score
)
from backend.core.llm_analyzer import analyze_with_llm
from backend.models.schemas import FullAnalysisResult, SkillExtractionResult, SkillComparisonResult, FinalScore

logger = logging.getLogger(__name__)


def run_full_analysis(pdf_path: str, jd_text: str) -> FullAnalysisResult:
    """
    Master pipeline function. Runs all phases in sequence and
    returns a complete FullAnalysisResult object.

    This is the single function the API calls — it orchestrates
    everything and returns one clean result.
    """
    logger.info("Starting full resume analysis pipeline...")

    # ── Phase 2: PDF Parsing ──────────────────────────────────
    logger.info("Step 1/5: Parsing PDF...")
    pdf_result = extract_text_from_pdf(pdf_path)
    resume_text = pdf_result["full_text"]
    sections = extract_sections(resume_text)

    # ── Phase 3: Skill Extraction ─────────────────────────────
    logger.info("Step 2/5: Extracting skills...")
    resume_skills_raw = extract_skills(resume_text)
    jd_skills_raw = extract_skills(jd_text)
    comparison_raw = compare_skills(resume_skills_raw, jd_skills_raw)

    resume_skills = SkillExtractionResult(**resume_skills_raw)
    jd_skills = SkillExtractionResult(**jd_skills_raw)
    skill_comparison = SkillComparisonResult(**comparison_raw)

    # ── Phase 4: Similarity ───────────────────────────────────
    logger.info("Step 3/5: Computing similarity scores...")
    semantic_score = compute_overall_semantic_score(resume_text, jd_text)
    section_scores = compute_section_similarities(sections, jd_text)
    final_raw = compute_final_score(
        keyword_match_pct=comparison_raw["match_percentage"],
        semantic_score=semantic_score,
        section_scores=section_scores
    )
    final_score = FinalScore(**final_raw)

    skill_gap_details = compute_skill_gap_embeddings(
        comparison_raw["missing_skills"],
        resume_text
    )

    # ── Phase 5: LLM Analysis ─────────────────────────────────
    logger.info("Step 4/5: Running LLM analysis...")
    llm_analysis = analyze_with_llm(
        resume_text=resume_text,
        jd_text=jd_text,
        matched_skills=comparison_raw["matched_skills"],
        missing_skills=comparison_raw["missing_skills"],
        semantic_score=semantic_score,
        keyword_match_pct=comparison_raw["match_percentage"],
        final_score=final_raw["final_score"]
    )

    # ── Assemble Final Result ─────────────────────────────────
    logger.info("Step 5/5: Assembling result...")
    result = FullAnalysisResult(
        resume_char_count=pdf_result["char_count"],
        resume_page_count=pdf_result["page_count"],
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        skill_comparison=skill_comparison,
        semantic_score=round(semantic_score, 4),
        section_scores=section_scores,
        final_score=final_score,
        llm_analysis=llm_analysis,
        skill_gap_details=skill_gap_details
    )

    logger.info(f"Analysis complete. Final score: {final_raw['final_score']}")
    return result