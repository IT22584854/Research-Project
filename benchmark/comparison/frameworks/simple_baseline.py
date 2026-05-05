from evaluation.metrics import semantic_similarity


def evaluate(question, context, answer, ground_truth):

    results = {}

    results["semantic_similarity"] = semantic_similarity(
        answer,
        ground_truth
    )

    return results