from sentence_transformers import SentenceTransformer
import numpy as np

# Load evaluation embedding model
eval_model = SentenceTransformer("intfloat/multilingual-e5-large")


def _format_e5(texts, prefix="passage"):
    """
    E5 requires query/passage prefixes.
    """

    formatted = []

    for t in texts:
        formatted.append(f"{prefix}: {t}")

    return formatted


def embed_texts(texts, batch_size=16, prefix="passage"):

    texts = _format_e5(texts, prefix)

    embeddings = eval_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True
    )

    return np.array(embeddings)


def embed(text, prefix="passage"):
    """
    Compatible function used by evaluation metrics.
    """

    if isinstance(text, str):
        text = [text]

    text = _format_e5(text, prefix)

    embedding = eval_model.encode(
        text,
        batch_size=1,
        normalize_embeddings=True
    )

    return np.array(embedding)