import sys
sys.path.insert(0, ".")

from backend.core.pdf_parser import extract_text_from_pdf, extract_sections
from backend.core.extractor import extract_skills, compare_skills
from backend.core.similarity import (
    compute_overall_semantic_score,
    compute_section_similarities,
    compute_skill_gap_embeddings,
    compute_final_score
)
from backend.core.llm_analyzer import analyze_with_llm

SAMPLE_JD = """
We are looking for a Python Backend Developer.
Requirements:
- 1-2 years of experience with Python and FastAPI or Django
- Strong knowledge of SQL and PostgreSQL
- Experience with REST APIs and Git
- Familiarity with Docker and AWS is a plus
- Understanding of Data Structures and Algorithms
- Knowledge of Machine Learning or NLP is a bonus
Education: B.Tech or B.E in any engineering discipline
"""

def test_full_pipeline():
    print("Running full pipeline test...\n")

    # Step 1: Parse PDF
    result = extract_text_from_pdf("sample_data/sample_resume.pdf")
    resume_text = result["full_text"]
    sections = extract_sections(resume_text)

    # Step 2: Extract skills
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(SAMPLE_JD)
    comparison = compare_skills(resume_skills, jd_skills)

    # Step 3: Similarity
    semantic_score = compute_overall_semantic_score(resume_text, SAMPLE_JD)
    section_scores = compute_section_similarities(sections, SAMPLE_JD)
    final = compute_final_score(
        keyword_match_pct=comparison["match_percentage"],
        semantic_score=semantic_score,
        section_scores=section_scores
    )

    print(f"Keyword match : {comparison['match_percentage']}%")
    print(f"Semantic score: {semantic_score}")
    print(f"Final score   : {final['final_score']} — {final['label']}")
    print(f"Missing skills: {comparison['missing_skills']}")

    # Step 4: LLM Analysis
    print("\nCalling Gemini API...")
    analysis = analyze_with_llm(
        resume_text=resume_text,
        jd_text=SAMPLE_JD,
        matched_skills=comparison["matched_skills"],
        missing_skills=comparison["missing_skills"],
        semantic_score=semantic_score,
        keyword_match_pct=comparison["match_percentage"],
        final_score=final["final_score"]
    )

    print("\n" + "=" * 60)
    print("GEMINI ANALYSIS RESULT")
    print("=" * 60)
    print(f"\nOverall Assessment:\n{analysis.overall_assessment}")
    print(f"\nTop Strengths:")
    for s in analysis.top_strengths:
        print(f"  + {s}")
    print(f"\nCritical Gaps:")
    for g in analysis.critical_gaps:
        print(f"  - {g}")
    print(f"\nImprovement Suggestions:")
    for imp in analysis.improvement_suggestions:
        print(f"  [{imp.priority.upper()}] {imp.section}: {imp.suggestion}")
    print(f"\nATS Keywords to Add: {analysis.ats_keywords_to_add}")
    print(f"\nRewritten Summary:\n{analysis.rewritten_summary}")
    print(f"\nHiring Probability: {analysis.hiring_probability.upper()}")

if __name__ == "__main__":
    test_full_pipeline()
