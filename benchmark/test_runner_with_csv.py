import json
import yaml
import csv
from utils.markdown_loader import load_corpus
from src.metrics.factual_accuracy import factual_accuracy
from src.metrics.semantic_similarity import semantic_similarity
from src.metrics.safety import safety_score
from src.metrics.completeness import completeness
from src.metrics.source_attribution import source_attribution
from src.profile_selector import select_profile
from src.scorer import compute_weighted_score
from src.grader import grade

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load corpus
corpus = load_corpus("data/corpus_testing")

# Load gold questions
with open("data/benchmarks/gold_questions.json") as f:
    questions = {q["id"]: q for q in json.load(f)}

# Load agent responses
with open("data/benchmarks/mock_agent_responses.json") as f:
    responses = json.load(f)

results = []

for item in responses:
    q = questions[item["question_id"]]
    response = item["response"]

    metrics = {
        "factual_accuracy": factual_accuracy(response, corpus),
        "semantic_similarity": semantic_similarity(q["question"], response),
        "safety": safety_score(response),
        "completeness": completeness(response, q["expected_points"]),
        "source_attribution": source_attribution(response)
    }

    profile = select_profile(q["category"], q["has_ground_truth"])
    score = compute_weighted_score(metrics, profile, "config.yaml")
    grade_label = grade(score, config["thresholds"], metrics["safety"])

    results.append({
        "question_id": item["question_id"],
        "category": q["category"],
        "profile": profile,
        **metrics,
        "final_score": score,
        "grade": grade_label
    })

# import to csv
output_path = "outputs/evaluation_results.csv"
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults exported to {output_path}")
