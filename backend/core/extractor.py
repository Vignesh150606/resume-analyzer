import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# MASTER SKILL DICTIONARY
# This is your ground truth. The extractor looks for these exact
# terms in the text. Organized by category for structured output.
# You can always add more skills to any category.
# ---------------------------------------------------------------

SKILL_TAXONOMY = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "go", "golang", "rust", "swift", "kotlin", "ruby", "php", "scala",
        "r", "matlab", "bash", "shell", "perl", "dart", "elixir"
    ],
    "web_frontend": [
        "html", "css", "react", "reactjs", "angular", "vue", "vuejs",
        "nextjs", "next.js", "tailwind", "bootstrap", "sass", "redux",
        "webpack", "vite", "jquery"
    ],
    "web_backend": [
        "fastapi", "flask", "django", "express", "expressjs", "nodejs",
        "node.js", "spring", "springboot", "spring boot", "laravel",
        "rails", "ruby on rails", "asp.net", "fastify"
    ],
    "databases": [
        "sql", "mysql", "postgresql", "postgres", "sqlite", "mongodb",
        "redis", "cassandra", "dynamodb", "oracle", "firebase",
        "elasticsearch", "neo4j", "supabase"
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "k8s", "terraform", "ansible", "jenkins", "github actions",
        "ci/cd", "linux", "nginx", "apache"
    ],
    "ai_ml": [
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "pandas", "numpy", "matplotlib", "huggingface", "langchain",
        "openai", "gemini", "llm", "transformers", "bert", "gpt",
        "neural network", "random forest", "xgboost"
    ],
    "tools_practices": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence",
        "postman", "swagger", "rest", "restful", "graphql", "grpc",
        "microservices", "agile", "scrum", "tdd", "unit testing",
        "pytest", "junit", "linux", "vs code", "docker"
    ],
    "core_cs": [
        "data structures", "algorithms", "object oriented programming",
        "oop", "design patterns", "system design", "operating systems",
        "computer networks", "dbms", "oops", "recursion", "dynamic programming"
    ]
}

# Flatten for quick lookup — maps lowercase skill -> category
SKILL_TO_CATEGORY = {}
for category, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        SKILL_TO_CATEGORY[skill.lower()] = category


def extract_skills(text: str) -> dict:
    """
    Extract skills from text by matching against the skill taxonomy.

    Returns:
        {
            "all_skills": ["python", "fastapi", ...],
            "by_category": {
                "programming_languages": ["python"],
                "web_backend": ["fastapi"],
                ...
            },
            "skill_count": 12
        }
    """
    if not text:
        return {"all_skills": [], "by_category": {}, "skill_count": 0}

    text_lower = text.lower()
    found_skills = {}  # skill -> category

    for skill, category in SKILL_TO_CATEGORY.items():
        # Use word boundary matching to avoid partial matches
        # e.g., "c" should not match inside "react" or "science"
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills[skill] = category

    # Group by category
    by_category = {}
    for skill, category in found_skills.items():
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(skill)

    # Sort skills in each category alphabetically
    for category in by_category:
        by_category[category].sort()

    all_skills = sorted(found_skills.keys())

    return {
        "all_skills": all_skills,
        "by_category": by_category,
        "skill_count": len(all_skills)
    }


def extract_experience_years(text: str) -> Optional[int]:
    """
    Try to extract years of experience mentioned in a job description.
    Handles patterns like:
        - "3+ years of experience"
        - "minimum 2 years"
        - "at least 5 years"
    Returns the number found, or None if not mentioned.
    """
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*(?:relevant\s*)?experience',
        r'minimum\s*(\d+)\s*years?',
        r'at\s*least\s*(\d+)\s*years?',
        r'(\d+)\s*to\s*\d+\s*years?\s*of\s*experience',
    ]

    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return int(match.group(1))

    return None


def extract_education_requirements(text: str) -> list:
    """
    Extract education requirements from job description.
    """
    requirements = []
    text_lower = text.lower()

    degree_patterns = {
        "B.Tech/B.E": r'\b(b\.?tech|b\.?e\.?|bachelor.{0,20}engineer)\b',
        "B.Sc": r'\b(b\.?sc|bachelor.{0,20}science)\b',
        "M.Tech/M.E": r'\b(m\.?tech|m\.?e\.?|master.{0,20}engineer)\b',
        "MBA": r'\b(mba|master.{0,20}business)\b',
        "Any Graduate": r'\b(any graduate|any degree|bachelor.{0,10}degree)\b',
    }

    for degree_name, pattern in degree_patterns.items():
        if re.search(pattern, text_lower):
            requirements.append(degree_name)

    return requirements


def compare_skills(resume_skills: dict, jd_skills: dict) -> dict:
    """
    Compare skills extracted from resume vs job description.

    Returns:
        {
            "matched_skills": [...],
            "missing_skills": [...],
            "extra_skills": [...],
            "match_percentage": 75.0,
            "matched_by_category": {...},
            "missing_by_category": {...}
        }
    """
    resume_set = set(resume_skills["all_skills"])
    jd_set = set(jd_skills["all_skills"])

    matched = sorted(resume_set & jd_set)      # In both
    missing = sorted(jd_set - resume_set)      # In JD but not resume
    extra = sorted(resume_set - jd_set)        # In resume but not JD

    # Calculate match percentage
    if len(jd_set) == 0:
        match_percentage = 0.0
    else:
        match_percentage = round((len(matched) / len(jd_set)) * 100, 1)

    # Group matched and missing by category
    matched_by_category = {}
    for skill in matched:
        cat = SKILL_TO_CATEGORY.get(skill, "other")
        matched_by_category.setdefault(cat, []).append(skill)

    missing_by_category = {}
    for skill in missing:
        cat = SKILL_TO_CATEGORY.get(skill, "other")
        missing_by_category.setdefault(cat, []).append(skill)

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "match_percentage": match_percentage,
        "matched_by_category": matched_by_category,
        "missing_by_category": missing_by_category
    }