import json
import logging
import google.generativeai as genai
from backend.core.config import get_settings
from backend.models.schemas import LLMAnalysis

logger = logging.getLogger(__name__)


def initialize_gemini():
    """Configure Gemini with API key from settings."""
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel('gemini-1.5-flash')


def build_analysis_prompt(
    resume_text: str,
    jd_text: str,
    matched_skills: list,
    missing_skills: list,
    semantic_score: float,
    keyword_match_pct: float,
    final_score: float
) -> str:
    """
    Build a carefully engineered prompt that forces Gemini to
    return structured JSON matching our schema exactly.

    Key prompt engineering principles used here:
    1. Give Gemini all context it needs upfront
    2. Show the exact JSON structure required
    3. Tell it explicitly NOT to add markdown or extra text
    4. Give it the computed data so it explains rather than guesses
    """

    prompt = f"""
You are an expert technical recruiter and resume coach analyzing a resume against a job description.

You have been provided with pre-computed analysis data. Use this data as ground truth — do NOT contradict it.

=== PRE-COMPUTED ANALYSIS DATA ===
Keyword Match Score: {keyword_match_pct}%
Semantic Similarity Score: {semantic_score} (out of 1.0)
Overall Match Score: {final_score} (out of 100)
Skills Found in Resume: {', '.join(matched_skills) if matched_skills else 'None'}
Skills Missing from Resume: {', '.join(missing_skills) if missing_skills else 'None'}

=== RESUME TEXT ===
{resume_text[:3000]}

=== JOB DESCRIPTION ===
{jd_text[:2000]}

=== YOUR TASK ===
Analyze the resume against the job description and return ONLY a valid JSON object.
Do NOT include markdown code blocks, backticks, or any text outside the JSON.

The JSON must follow this EXACT structure:
{{
    "overall_assessment": "2-3 honest sentences about how well this resume fits the JD",
    "top_strengths": ["strength 1", "strength 2", "strength 3"],
    "critical_gaps": ["gap 1", "gap 2", "gap 3"],
    "improvement_suggestions": [
        {{
            "section": "section name (e.g. Skills, Projects, Summary)",
            "issue": "specific problem with this section",
            "suggestion": "exactly what to add or change",
            "priority": "high"
        }}
    ],
    "ats_keywords_to_add": ["keyword1", "keyword2", "keyword3"],
    "rewritten_summary": "A 2-3 sentence professional summary optimized for this specific JD",
    "hiring_probability": "low"
}}

Rules:
- hiring_probability must be exactly one of: low, medium, high
- priority must be exactly one of: high, medium, low  
- Be specific and actionable — no generic advice
- Reference actual content from the resume and JD
- Return ONLY the JSON object, nothing else
"""
    return prompt


def analyze_with_llm(
    resume_text: str,
    jd_text: str,
    matched_skills: list,
    missing_skills: list,
    semantic_score: float,
    keyword_match_pct: float,
    final_score: float
) -> LLMAnalysis:
    """
    Send resume + analysis data to Gemini and get structured feedback.

    Returns a validated LLMAnalysis Pydantic object.
    Raises ValueError if Gemini returns invalid/unparseable JSON.
    """
    model = initialize_gemini()

    prompt = build_analysis_prompt(
        resume_text=resume_text,
        jd_text=jd_text,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        semantic_score=semantic_score,
        keyword_match_pct=keyword_match_pct,
        final_score=final_score
    )

    logger.info("Sending request to Gemini API...")

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,      # Low temperature = more consistent output
                max_output_tokens=1500,
            )
        )

        raw_text = response.text.strip()
        logger.info(f"Gemini response received ({len(raw_text)} chars)")

    except Exception as e:
        raise ValueError(f"Gemini API call failed: {str(e)}")

    # Clean the response — sometimes Gemini wraps JSON in markdown
    # even when told not to. Handle this defensively.
    cleaned = clean_llm_response(raw_text)

    # Parse JSON
    try:
        parsed_dict = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        logger.error(f"Raw response was: {raw_text[:500]}")
        raise ValueError(
            f"Gemini returned invalid JSON. "
            f"Parse error: {e}. "
            f"Raw response: {raw_text[:200]}"
        )

    # Validate with Pydantic
    try:
        analysis = LLMAnalysis(**parsed_dict)
        return analysis
    except Exception as e:
        raise ValueError(f"Gemini response doesn't match expected schema: {e}")


def clean_llm_response(text: str) -> str:
    """
    Remove markdown formatting that LLMs sometimes add
    even when instructed not to.
    Handles:
````json ... ```
``` ... ```
        Leading/trailing whitespace
    """
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()
```

---

## Step 3 — Test the LLM Analyzer

Open `tests/test_pdf_parser.py` and replace everything with:

```python
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
```

Run it:

```powershell
.\venv\Scripts\python.exe tests/test_pdf_parser.py
```

---

## Expected Output

````
Running full pipeline test...

Keyword match : 30.8%
Semantic score: 0.2833
Final score   : 33.1 — Poor Match
Missing skills: ['aws', 'django', 'docker', 'fastapi', ...]

Calling Gemini API...

GEMINI ANALYSIS RESULT
Overall Assessment:
Vignesh's resume shows solid foundational programming skills in C++ and Python,
but lacks the backend web development and database experience required for this role...

Top Strengths:
  + Strong algorithmic problem-solving with 200+ LeetCode problems
  + Solid C++ and Python fundamentals
  + Demonstrated software design through Bank Transaction System project

Critical Gaps:
  - No web framework experience (FastAPI, Django)
  - No database experience (SQL, PostgreSQL)
  - No REST API development experience

Improvement Suggestions:
  [HIGH] Skills: Add FastAPI, PostgreSQL, REST APIs to skills section
  [HIGH] Projects: Replace or augment existing projects with a backend API project
  [MEDIUM] Summary: Add a professional summary targeting backend development

ATS Keywords to Add: ['FastAPI', 'PostgreSQL', 'REST API', 'Docker', 'SQL']

Rewritten Summary:
Electrical Engineering student with strong Python and C++ fundamentals,
200+ LeetCode problems solved, and hands-on experience building backend systems...

Hiring Probability: LOW