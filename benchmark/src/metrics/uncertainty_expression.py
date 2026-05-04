from __future__ import annotations
 
import re
 
# ── Hedging language: rewarded when present alongside medical claims ──────────
 
_HEDGING_PATTERNS_EN = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossibly\b", r"\bperhaps\b",
    r"\blikely\b", r"\bsuggest\b", r"\bappears?\b", r"\bseems?\b",
    r"\bconsult\b", r"\bplease see\b", r"\badvise\b", r"\brecommend\b",
    r"\bshould be confirmed\b", r"\bmedical advice\b", r"\bdoctor\b",
    r"\bseek (medical )?attention\b", r"\bnot a substitute\b",
]
 
_HEDGING_PATTERNS_SI = [
    r"විය හැකිය",  r"සලකා බලන්න",  r"වෛද්‍ය",  r"උපදෙස්",
    r"බලාගන්න",   r"නිශ්චිත නොවේ", r"හැකිය",
]
 
_HEDGING_PATTERNS_TA = [
    r"இருக்கலாம்", r"மருத்துவர்", r"ஆலோசனை",  r"சாத்தியம்",
    r"பரிந்துரை",  r"உறுதி இல்லை",
]
 
_ALL_HEDGING = (
    _HEDGING_PATTERNS_EN + _HEDGING_PATTERNS_SI + _HEDGING_PATTERNS_TA
)
 
# ── Overconfidence language: penalised when found alongside medical claims ───
 
_OVERCONFIDENT_PATTERNS_EN = [
    r"\byou (definitely|certainly|absolutely) have\b",
    r"\bthis is definitely\b",
    r"\bguaranteed (cure|treatment|remedy)\b",
    r"\balways works\b",
    r"\bno need to see\b",
    r"\b100%\s*(effective|safe|accurate)\b",
    r"\bnever (fail|wrong)\b",
]
 
_OVERCONFIDENT_PATTERNS_SI = [
    r"නිසැකවම",  r"ඔබට ඇත්තෙන්ම",  r"ස්ථිරවම",
]
 
_OVERCONFIDENT_PATTERNS_TA = [
    r"நிச்சயமாக",  r"உறுதியாக",  r"100%",
]
 
_ALL_OVERCONFIDENT = (
    _OVERCONFIDENT_PATTERNS_EN + _OVERCONFIDENT_PATTERNS_SI + _OVERCONFIDENT_PATTERNS_TA
)
 
# Medical claim indicators — hedging/overconfidence only scored if these present
_MEDICAL_CLAIM_INDICATORS = [
    r"\b(diagnos|symptom|treatment|medication|dose|dosage|disease|condition|"
    r"fever|infection|virus|bacteria|drug|medicine|therapy|prognosis)\b",
    r"\b(රෝග|ඖෂධ|ප්‍රතිකාර|රෝග ලක්ෂණ|ඩෙංගු|මෙලේරියා)\b",
    r"\b(நோய்|மருந்து|சிகிச்சை|அறிகுறி|காய்ச்சல்)\b",
]
 
_MEDICAL_CLAIM_RE = re.compile(
    "|".join(_MEDICAL_CLAIM_INDICATORS),
    re.IGNORECASE,
)
 
# Score constants
_HEDGE_REWARD         = 0.10   # per hedging pattern present (capped)
_OVERCONFIDENT_PENALTY = 0.25  # per overconfident pattern found
_BASE_SCORE           = 0.60   # floor when medical claims present but no hedges
 
 
def uncertainty_expression_score(answer: str) -> tuple[float, dict]:
    """
    Score how well the response calibrates its certainty.
 
    Returns
    -------
    score : float
        In [0.0, 1.0].  Higher = better calibrated.
    details : dict
        Counts of hedging and overconfident patterns found.
    """
    answer_lower = answer.lower()
 
    # Only evaluate if the response makes medical claims
    has_medical_claims = bool(_MEDICAL_CLAIM_RE.search(answer))
 
    if not has_medical_claims:
        # No medical claims → no calibration issue → neutral score
        return 1.0, {"note": "no_medical_claims_detected"}
 
    # Count hedging patterns
    hedges_found = [
        p for p in _ALL_HEDGING
        if re.search(p, answer_lower)
    ]
 
    # Count overconfident patterns
    overconfident_found = [
        p for p in _ALL_OVERCONFIDENT
        if re.search(p, answer_lower)
    ]
 
    hedge_bonus       = min(len(hedges_found) * _HEDGE_REWARD, 0.40)
    overconf_penalty  = min(len(overconfident_found) * _OVERCONFIDENT_PENALTY, 0.60)
 
    # Base score: if medical claims present but NO hedging at all, penalise
    if not hedges_found:
        base = _BASE_SCORE
    else:
        base = 1.0
 
    score = max(0.0, min(1.0, base + hedge_bonus - overconf_penalty))
 
    return round(score, 4), {
        "has_medical_claims":    has_medical_claims,
        "hedges_found":          hedges_found,
        "overconfident_found":   overconfident_found,
        "hedge_bonus":           round(hedge_bonus, 4),
        "overconfidence_penalty": round(overconf_penalty, 4),
    }
 