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

def test_similarity():
    print("Loading model... (first time takes 10-15 seconds)")
    
    # Load resume
    result = extract_text_from_pdf("sample_data/sample_resume.pdf")
    resume_text = result["full_text"]
    sections = extract_sections(resume_text)
    
    # Get skills
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(SAMPLE_JD)
    comparison = compare_skills(resume_skills, jd_skills)

    print("\n" + "=" * 60)
    print("TEST 1: Overall semantic similarity")
    print("=" * 60)
    semantic_score = compute_overall_semantic_score(resume_text, SAMPLE_JD)
    print(f"Semantic similarity score: {semantic_score}")
    print(f"(1.0 = perfect match, 0.0 = no relation)")

    print("\n" + "=" * 60)
    print("TEST 2: Section-by-section similarity")
    print("=" * 60)
    section_scores = compute_section_similarities(sections, SAMPLE_JD)
    for section, score in sorted(section_scores.items(),
                                  key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 20)
        print(f"  {section:<25} {score:.4f}  {bar}")

    print("\n" + "=" * 60)
    print("TEST 3: Skill gap semantic analysis")
    print("=" * 60)
    gap_analysis = compute_skill_gap_embeddings(
        comparison["missing_skills"],
        resume_text
    )
    for item in gap_analysis:
        status = "~ Possibly familiar" if item["likely_known"] else "✗ Likely missing"
        print(f"  {item['skill']:<20} score={item['semantic_score']}  {status}")

    print("\n" + "=" * 60)
    print("TEST 4: Final combined score")
    print("=" * 60)
    final = compute_final_score(
        keyword_match_pct=comparison["match_percentage"],
        semantic_score=semantic_score,
        section_scores=section_scores
    )
    print(f"  Final Score : {final['final_score']} / 100")
    print(f"  Grade       : {final['grade']} — {final['label']}")
    print(f"  Breakdown   : {final['breakdown']}")

if __name__ == "__main__":
    test_similarity()