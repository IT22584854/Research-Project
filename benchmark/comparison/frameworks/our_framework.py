from evaluation.metrics import (
    semantic_similarity,
    groundedness,
    safety_score
)

def evaluate(question, context, answer, ground_truth):

    results = {}

    results["semantic_similarity"] = semantic_similarity(
        answer, ground_truth
    )

    results["groundedness"] = groundedness(
        answer, context
    )

    results["safety"] = safety_score(answer)

    return results