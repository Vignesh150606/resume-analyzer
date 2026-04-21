import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# MODEL LOADING
# We use 'all-MiniLM-L6-v2' — the best balance of speed and
# accuracy for semantic similarity tasks. It produces 384-
# dimensional embeddings and runs fast even on CPU.
# lru_cache ensures the model loads only ONCE — loading a
# transformer model takes 2-3 seconds, we don't want that
# happening on every request.
# ---------------------------------------------------------------

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Load and cache the sentence transformer model.
    First call takes 2-3 seconds. All subsequent calls are instant.
    """
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Model loaded successfully.")
    return model


def get_embedding(text: str) -> np.ndarray:
    """
    Convert text into a numerical embedding vector.
    
    Args:
        text: Any string of text
        
    Returns:
        numpy array of shape (384,) — 384 numbers representing
        the semantic meaning of the text
    """
    model = get_model()
    # encode() returns a numpy array
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def compute_cosine_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two pieces of text.
    
    Returns a float between 0.0 and 1.0.
    1.0 = identical meaning, 0.0 = completely unrelated.
    """
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    
    # sklearn expects 2D arrays, so we reshape from (384,) to (1, 384)
    emb1_2d = emb1.reshape(1, -1)
    emb2_2d = emb2.reshape(1, -1)
    
    score = cosine_similarity(emb1_2d, emb2_2d)[0][0]
    
    # Convert from numpy float to Python float for JSON serialization
    return float(round(score, 4))


def compute_section_similarities(resume_sections: dict, jd_text: str) -> dict:
    """
    Compare each resume section against the full job description.
    This tells us which parts of the resume are most relevant
    to the job and which are weak.
    
    Returns:
        {
            "section_name": similarity_score,
            ...
        }
    """
    section_scores = {}
    
    for section_name, section_content in resume_sections.items():
        if not section_content or not section_content.strip():
            continue
        if len(section_content.split()) < 3:
            # Skip sections with less than 3 words — not meaningful
            continue
            
        score = compute_cosine_similarity(section_content, jd_text)
        section_scores[section_name] = score
        logger.info(f"Section '{section_name}' similarity: {score}")
    
    return section_scores


def compute_overall_semantic_score(resume_text: str, jd_text: str) -> float:
    """
    Compute overall semantic similarity between full resume and JD.
    """
    return compute_cosine_similarity(resume_text, jd_text)


def compute_skill_gap_embeddings(
    missing_skills: list,
    resume_text: str
) -> list:
    """
    For each missing skill, compute how semantically close the resume
    is to that skill — even if the exact keyword isn't present.
    
    This catches cases like: JD says "NumPy" but resume says
    "numerical computing with arrays" — semantically related
    even though keyword matching would miss it.
    
    Returns list of dicts sorted by similarity (highest first):
        [{"skill": "numpy", "semantic_score": 0.72, "likely_known": True}, ...]
    """
    if not missing_skills:
        return []
    
    results = []
    
    for skill in missing_skills:
        score = compute_cosine_similarity(skill, resume_text)
        results.append({
            "skill": skill,
            "semantic_score": float(round(score, 4)),
            # If score > 0.3, the resume has related content
            # even without the exact keyword
            "likely_known": score > 0.3
        })
    
    # Sort by score descending — most related missing skills first
    results.sort(key=lambda x: x["semantic_score"], reverse=True)
    
    return results


def compute_final_score(
    keyword_match_pct: float,
    semantic_score: float,
    section_scores: dict
) -> dict:
    """
    Combine keyword matching + semantic similarity into one final score.
    
    Weighting rationale:
    - Keyword match (40%): Hard requirements — specific tools/languages
    - Semantic similarity (40%): Overall relevance of experience
    - Best section score (20%): Reward strong relevant sections
    
    Returns score out of 100 with a letter grade.
    """
    # Get the best performing section score
    best_section = max(section_scores.values()) if section_scores else 0.0
    
    # Weighted combination
    # keyword_match_pct is already 0-100, convert others to 0-100
    final = (
        (keyword_match_pct * 0.40) +
        (semantic_score * 100 * 0.40) +
        (best_section * 100 * 0.20)
    )
    
    final = round(final, 1)
    
    # Assign letter grade
    if final >= 80:
        grade = "A"
        label = "Strong Match"
    elif final >= 65:
        grade = "B"
        label = "Good Match"
    elif final >= 50:
        grade = "C"
        label = "Moderate Match"
    elif final >= 35:
        grade = "D"
        label = "Weak Match"
    else:
        grade = "F"
        label = "Poor Match"
    
    return {
        "final_score": final,
        "grade": grade,
        "label": label,
        "breakdown": {
            "keyword_match_contribution": round(keyword_match_pct * 0.40, 1),
            "semantic_contribution": round(semantic_score * 100 * 0.40, 1),
            "section_contribution": round(best_section * 100 * 0.20, 1)
        }
    }