import json
import logging
import google.generativeai as genai
from backend.core.config import get_settings
from backend.models.schemas import LLMAnalysis, ImprovementSuggestion

logger = logging.getLogger(__name__)


def initialize_gemini():
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel('gemini-1.5-flash-latest')


def build_analysis_prompt(
    resume_text: str,
    jd_text: str,
    matched_skills: list,
    missing_skills: list,
    semantic_score: float,
    keyword_match_pct: float,
    final_score: float
) -> str:
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


def clean_llm_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def analyze_with_llm(
    resume_text: str,
    jd_text: str,
    matched_skills: list,
    missing_skills: list,
    semantic_score: float,
    keyword_match_pct: float,
    final_score: float
) -> LLMAnalysis:

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
                temperature=0.3,
                max_output_tokens=1500,
            )
        )
        raw_text = response.text.strip()
        logger.info(f"Gemini response received ({len(raw_text)} chars)")

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "404" in error_str or "quota" in error_str.lower() or "exhausted" in error_str.lower() or "not found" in error_str.lower():
            logger.warning("Gemini quota exceeded — returning fallback analysis")
            return _fallback_analysis(missing_skills, keyword_match_pct)
        raise ValueError(f"Gemini API call failed: {error_str}")

    # Clean and parse
    cleaned = clean_llm_response(raw_text)

    try:
        parsed_dict = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        logger.error(f"Raw response was: {raw_text[:500]}")
        raise ValueError(f"Gemini returned invalid JSON: {e}. Raw: {raw_text[:200]}")

    try:
        analysis = LLMAnalysis(**parsed_dict)
        return analysis
    except Exception as e:
        raise ValueError(f"Gemini response doesn't match expected schema: {e}")


def _fallback_analysis(missing_skills: list, keyword_match_pct: float) -> LLMAnalysis:
    if keyword_match_pct >= 70:
        probability = "high"
        assessment = "Strong keyword match detected. Resume aligns well with the job requirements."
    elif keyword_match_pct >= 40:
        probability = "medium"
        assessment = "Moderate match. Resume covers some requirements but has notable gaps."
    else:
        probability = "low"
        assessment = "Weak match. Resume is missing several key skills required for this role."

    suggestions = []
    for skill in missing_skills[:3]:
        suggestions.append(ImprovementSuggestion(
            section="Skills",
            issue=f"Missing required skill: {skill}",
            suggestion=f"Add {skill} to your skills section and build a project using it",
            priority="high"
        ))

    if not suggestions:
        suggestions.append(ImprovementSuggestion(
            section="Projects",
            issue="Projects could be more aligned with target role",
            suggestion="Add projects that directly use technologies mentioned in the JD",
            priority="medium"
        ))

    return LLMAnalysis(
        overall_assessment=assessment,
        top_strengths=[
            "Problem solving ability",
            "Programming fundamentals",
            "Willingness to learn"
        ],
        critical_gaps=missing_skills[:3] if missing_skills else ["No critical gaps detected"],
        improvement_suggestions=suggestions,
        ats_keywords_to_add=missing_skills[:5],
        rewritten_summary="Motivated engineering graduate with strong programming fundamentals seeking to apply technical skills in a software development role.",
        hiring_probability=probability
    )