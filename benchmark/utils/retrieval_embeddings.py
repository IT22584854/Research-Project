from FlagEmbedding import BGEM3FlagModel
import numpy as np

# Initialize retrieval model once
bge_model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)


def embed_texts(texts, batch_size=16):
    """
    Batch embedding for indexing or retrieval.
    Uses BGE-M3.
    """

    all_embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        embeddings = bge_model.encode(
            batch,
            batch_size=batch_size,
            max_length=512
        )["dense_vecs"]

        all_embeddings.append(embeddings)

        print(f"Embedded {i + len(batch)} / {len(texts)}")

    return np.vstack(all_embeddings)


def embed(text):
    """
    Single-text embedding for retrieval.
    """

    if isinstance(text, str):
        text = [text]

    embedding = bge_model.encode(
        text,
        batch_size=1,
        max_length=512
    )["dense_vecs"]

    return np.array(embedding)