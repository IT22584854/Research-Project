from FlagEmbedding import BGEM3FlagModel
import numpy as np
import torch

# ----------------------------
# DEVICE SETUP
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# LOAD BGE-M3 MODEL (load once)
# ----------------------------
model = BGEM3FlagModel(
    'BAAI/bge-m3',
    use_fp16=(device == "cuda")  # safe fallback for CPU
)

# ----------------------------
# OPTIONAL: PREFIXES (boost retrieval quality)
# ----------------------------
DOC_PREFIX = "Represent this document for retrieval: "
QUERY_PREFIX = "Represent this query for searching relevant documents: "

# ----------------------------
# NORMALIZATION (important for cosine similarity)
# ----------------------------
def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)

# ----------------------------
# EMBED BATCH TEXTS (FOR INDEXING)
# ----------------------------
def embed_texts(texts, batch_size=16, use_prefix=True):
    """
    Convert list of texts into embeddings for Pinecone indexing.
    """

    all_embeddings = []

    # add prefix (optional but recommended for BGE-M3)
    if use_prefix:
        texts = [DOC_PREFIX + t for t in texts]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        output = model.encode(
            batch,
            batch_size=batch_size,
            max_length=512
        )

        embeddings = output["dense_vecs"]

        # normalize for cosine similarity
        embeddings = _normalize(embeddings)

        all_embeddings.append(embeddings)

        print(f"✅ Embedded {min(i + len(batch), len(texts))} / {len(texts)}")

    return np.vstack(all_embeddings).tolist()

# ----------------------------
# SINGLE TEXT EMBEDDING (FOR QUERY / EVAL)
# ----------------------------
def embed(text, is_query=True):
    """
    Embed a single text (used for search queries or evaluation).
    """

    if isinstance(text, str):
        text = [text]

    if is_query:
        text = [QUERY_PREFIX + t for t in text]

    output = model.encode(
        text,
        batch_size=1,
        max_length=512
    )

    embedding = output["dense_vecs"]

    embedding = _normalize(embedding)

    return embedding[0].tolist()