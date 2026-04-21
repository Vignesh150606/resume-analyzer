import sys
sys.path.insert(0, ".")

from backend.core.pdf_parser import extract_text_from_pdf
from backend.core.extractor import extract_skills, compare_skills, extract_experience_years

# Sample job description for testing
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

def test_extractor():
    print("=" * 60)
    print("TEST 1: Extract skills from YOUR resume")
    print("=" * 60)

    result = extract_text_from_pdf("sample_data/sample_resume.pdf")
    resume_skills = extract_skills(result["full_text"])

    print(f"Total skills found in resume: {resume_skills['skill_count']}")
    print(f"All skills: {resume_skills['all_skills']}")
    print(f"\nBy category:")
    for cat, skills in resume_skills["by_category"].items():
        print(f"  {cat}: {skills}")

    print("\n")
    print("=" * 60)
    print("TEST 2: Extract skills from Job Description")
    print("=" * 60)

    jd_skills = extract_skills(SAMPLE_JD)
    print(f"Total skills in JD: {jd_skills['skill_count']}")
    print(f"JD requires: {jd_skills['all_skills']}")

    exp_years = extract_experience_years(SAMPLE_JD)
    print(f"Experience required: {exp_years} years")

    print("\n")
    print("=" * 60)
    print("TEST 3: Compare resume vs JD")
    print("=" * 60)

    comparison = compare_skills(resume_skills, jd_skills)
    print(f"Match percentage : {comparison['match_percentage']}%")
    print(f"Matched skills   : {comparison['matched_skills']}")
    print(f"Missing skills   : {comparison['missing_skills']}")
    print(f"Extra skills     : {comparison['extra_skills']}")

if __name__ == "__main__":
    test_extractor()
    