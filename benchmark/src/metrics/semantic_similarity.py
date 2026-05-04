from sklearn.metrics.pairwise import cosine_similarity
from utils.eval_embeddings import embed


def semantic_similarity(
    question: str,
    answer: str,
    retrieved_chunks: list[str] | None = None,
) -> dict:
    """
    Compute three complementary similarity signals.

    The original implementation produced a single Q→A cosine score.  This
    conflated two distinct quality dimensions:

    1. **Retrieval relevance** — are the retrieved chunks actually about what
       the user asked?  A retriever returning off-topic passages will produce
       a low-quality answer regardless of generation quality.

    2. **Generation faithfulness** — does the answer stay within the semantic
       space of the evidence?  A high Q→A score on a wrong answer just means
       the model paraphrased the question back convincingly.

    Separating the two lets the weighted scorer and W&B dashboards surface
    *where* quality broke down (retrieval vs generation).

    Parameters
    ----------
    question : str
    answer : str
    retrieved_chunks : list[str] | None
        If provided, retrieval-relevance and answer-grounding scores are
        also computed.  Pass ``None`` (or omit) to get only the Q→A score.

    Returns
    -------
    dict with keys:
        ``qa_similarity``          – cosine sim between question and answer
        ``retrieval_relevance``    – mean max cosine sim of chunks to question
                                     (None if no chunks supplied)
        ``answer_context_overlap`` – mean max cosine sim of answer to chunks
                                     (None if no chunks supplied)
    """

    # --- Q → A ---------------------------------------------------------------
    q_vec = embed([question], prefix="query")    # (1, dim)
    a_vec = embed([answer],   prefix="passage")  # (1, dim)
    qa_sim = float(cosine_similarity(q_vec, a_vec)[0][0])

    if not retrieved_chunks:
        return {
            "qa_similarity":          round(qa_sim, 4),
            "retrieval_relevance":    None,
            "answer_context_overlap": None,
        }

    chunk_vecs = embed(retrieved_chunks)  # (n_chunks, dim)

    # --- Retrieval relevance: Q → chunks -------------------------------------
    # For each chunk, how similar is it to the question?
    # We take the mean of the per-chunk maximum to get a corpus-level score.
    q_chunk_sims  = cosine_similarity(q_vec, chunk_vecs)[0]   # (n_chunks,)
    retrieval_rel = float(q_chunk_sims.mean())

    # --- Answer–context overlap: A → chunks ----------------------------------
    # For each chunk, how similar is it to the answer?
    # High score = answer stays within the evidence space.
    a_chunk_sims  = cosine_similarity(a_vec, chunk_vecs)[0]   # (n_chunks,)
    answer_ctx    = float(a_chunk_sims.max())   # max: at least one chunk supports it

    return {
        "qa_similarity":          round(qa_sim,        4),
        "retrieval_relevance":    round(retrieval_rel,  4),
        "answer_context_overlap": round(answer_ctx,     4),
    }