import yaml
from src.grader import grade
from src.profile_selector import select_profile
from src.scorer import compute_weighted_score

CONFIG_PATH = "config.yaml"

def evaluate_agent_response(agent_response, question):

    # --- Step 1: Compute metrics ---
    # For testing purposes, we mock them using simple keyword matching
    expected_points = question.get("expected_points", [])
    text = agent_response.lower()

    metrics = {}

    # Factual accuracy: % of expected points mentioned
    if expected_points:
        hits = sum(1 for p in expected_points if p.lower() in text)
        metrics["factual_accuracy"] = hits / len(expected_points)
    else:
        metrics["factual_accuracy"] = 0.0

    # Semantic similarity: simple proxy using keyword overlap
    metrics["semantic_similarity"] = metrics["factual_accuracy"]

    # Safety: always 1 for mock, real system would check harmful patterns
    metrics["safety"] = 1.0

    # Completeness: fraction of expected points present
    metrics["completeness"] = metrics["factual_accuracy"]

    # Source attribution: 1 if "ministry" or "government" mentioned
    metrics["source_attribution"] = 1.0 if "ministry" in text or "government" in text else 0.0

    # --- Step 2: Select profile ---
    profile = select_profile(question.get("category"), question.get("has_ground_truth", False))

    # --- Step 3: Compute weighted score ---
    score = compute_weighted_score(metrics, profile, CONFIG_PATH)

    # --- Step 4: Get grade ---
    thresholds = yaml.safe_load(open(CONFIG_PATH))["thresholds"]
    grade_label = grade(score, thresholds, metrics["safety"])

    return score, grade_label, metrics
