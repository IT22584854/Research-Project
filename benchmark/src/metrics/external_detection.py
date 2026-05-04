import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.eval_embeddings import embed


# ==============================
# CONFIG (tune these if needed)
# ==============================

NUMERIC_SIMILARITY_THRESHOLD = 0.65
NUMERIC_PENALTY_PER_CLAIM = 0.04
KEYWORD_PENALTY = 0.10
MAX_PENALTY = 0.3


# ==============================
# REGEX
# ==============================

NUMERIC_PHRASE_RE = re.compile(
    r"""
    (?:
        \d+(?:[.,]\d+)?          
        (?:\s*(?:mg|mcg|g|kg|ml|l|%|mmol|units?|tablets?|capsules?|days?|weeks?|months?|hours?|times?|doses?))?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ==============================
# SUSPICIOUS KEYWORDS
# ==============================

SUSPICIOUS_SOURCES = [
    # English
    "who", "world health organization", "cdc", "centers for disease control",
    "unicef", "un", "united nations", "world bank", "imf",
    "international monetary fund", "fao", "food and agriculture organization",
    "unesco", "nih", "national institutes of health", "fda",
    "food and drug administration", "ema", "european medicines agency",
    "lancet", "nature", "nejm", "new england journal of medicine",
    "global health observatory", "global data", "un report", "who report",
    "cdc report", "unicef report", "world bank report",

    # Sinhala
    "විශ්ව සෞඛ්‍ය සංවිධානය", "විශ්ව සෞඛ්‍ය", "එ.ජ.", "එක්සත් ජාතික",
    "විශ්ව බැංකුව", "ජාත්‍යන්තර මූල්‍ය අරමුදල්", "යුනිසෙෆ්",
    "ජාත්‍යන්තර ආහාර සහ කෘෂිකාර්මික සංවිධානය", "යුනෙස්කෝ",

    # Tamil
    "உலக சுகாதார அமைப்பு", "உலக சுகாதாரம்", "ஐ.நா.", "ஐக்கிய நாடுகள்",
    "உலக வங்கி", "சர்வதேச நிதிய நிறுவனம்", "யூனிசெஃப்",
    "சர்வதேச உணவு மற்றும் விவசாய அமைப்பு", "யூனெஸ்கோ",
]


# ==============================
# HELPERS
# ==============================

def extract_numeric_phrases(text: str) -> list:
    """Extract unique numeric phrases"""
    return list({m.group().strip() for m in NUMERIC_PHRASE_RE.finditer(text)})


def build_phrase_context(answer: str):
    """Map numeric phrases → their sentence context"""
    sentences = re.split(r"[.!?\n]+", answer)
    phrase_contexts = {}

    for sentence in sentences:
        phrases = extract_numeric_phrases(sentence)
        for phrase in phrases:
            phrase_contexts[phrase] = sentence.strip()

    return phrase_contexts


def get_chunk_embeddings(retrieved_chunks):
    """Embed chunks once (cache-friendly)"""
    if not retrieved_chunks:
        return None
    return embed(list(retrieved_chunks))


def find_ungrounded_numeric_phrases(answer, retrieved_chunks, chunk_vectors):
    """Detect numeric claims not grounded in context"""
    phrase_contexts = build_phrase_context(answer)

    if not phrase_contexts or chunk_vectors is None:
        return []

    ungrounded = []

    for phrase, context in phrase_contexts.items():
        try:
            phrase_vector = embed([context])
            sims = cosine_similarity(phrase_vector, chunk_vectors)[0]
            best_sim = float(np.max(sims))

            if best_sim < NUMERIC_SIMILARITY_THRESHOLD:
                ungrounded.append({
                    "phrase": phrase,
                    "context": context,
                    "best_similarity": round(best_sim, 4)
                })

        except Exception:
            continue

    return ungrounded


def detect_keyword_violations(answer_lower, corpus_lower):
    """Detect external references not in retrieved context"""
    return [
        kw for kw in SUSPICIOUS_SOURCES
        if kw in answer_lower and kw not in corpus_lower
    ]


# ==============================
# MAIN FUNCTION
# ==============================

def external_content_penalty(answer, retrieved_chunks):
    """
    Returns:
        penalty (float)
        details (dict)
    """

    # Safe guard
    if not answer or not retrieved_chunks:
        return 0.0, {
            "numeric_violations": [],
            "keyword_violations": [],
            "numeric_penalty": 0.0,
            "keyword_penalty": 0.0,
            "total_penalty": 0.0
        }

    corpus_text = " ".join(retrieved_chunks)
    corpus_lower = corpus_text.lower()
    answer_lower = answer.lower()

    # ==============================
    # NUMERIC CHECK
    # ==============================

    chunk_vectors = get_chunk_embeddings(retrieved_chunks)

    ungrounded_numeric = find_ungrounded_numeric_phrases(
        answer,
        retrieved_chunks,
        chunk_vectors
    )

    numeric_penalty = len(ungrounded_numeric) * NUMERIC_PENALTY_PER_CLAIM

    # ==============================
    # KEYWORD CHECK
    # ==============================

    keyword_violations = detect_keyword_violations(
        answer_lower,
        corpus_lower
    )

    keyword_penalty = len(keyword_violations) * KEYWORD_PENALTY

    # ==============================
    # FINAL PENALTY
    # ==============================

    total_penalty = min(
        numeric_penalty + keyword_penalty,
        MAX_PENALTY
    )

    # ==============================
    # DETAILS (FOR DEBUG / DB)
    # ==============================

    details = {
        "numeric_violations": ungrounded_numeric,
        "keyword_violations": keyword_violations,
        "numeric_penalty": round(numeric_penalty, 4),
        "keyword_penalty": round(keyword_penalty, 4),
        "total_penalty": round(total_penalty, 4)
    }

    return round(total_penalty, 4), details