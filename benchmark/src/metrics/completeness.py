import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.embeddings import embed


def completeness(answer, retrieved_chunks, threshold=0.6):

    # Safety check
    if not retrieved_chunks:
        return 0.0

    # Split chunks into sentences
    sentences = []
    for chunk in retrieved_chunks:
        sentences.extend(chunk.split("."))

    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    # Embed sentences and answer
    sentence_vecs = embed(sentences)
    answer_vec = embed([answer])

    # Compute similarity
    similarities = cosine_similarity(sentence_vecs, answer_vec)

    # Count covered facts
    covered = np.sum(similarities >= threshold)

    completeness_score = covered / len(sentences)

    return float(completeness_score)