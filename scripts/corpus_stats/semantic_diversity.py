from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_DIR = Path(r"D:\SL_Medical_Corpus\data\3_agent_export\markdown")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_DOC_CHARS = 4000
MAX_DOCS = 500  # increase if you want, but 500 is a good sample

doc_ids = []
texts = []

for fp in sorted(INPUT_DIR.glob("*.md")):
    text = fp.read_text(encoding="utf-8", errors="ignore").strip()
    if len(text) < 100:
        continue
    doc_ids.append(fp.stem)
    texts.append(text[:MAX_DOC_CHARS])

if len(texts) > MAX_DOCS:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(texts), size=MAX_DOCS, replace=False)
    texts = [texts[i] for i in idx]
    doc_ids = [doc_ids[i] for i in idx]

model = SentenceTransformer(MODEL_NAME)
emb = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
emb = np.asarray(emb)

K = emb @ emb.T

mask = ~np.eye(K.shape[0], dtype=bool)
off_diag = K[mask]

mean_similarity = float(off_diag.mean()) if off_diag.size else 0.0
std_similarity = float(off_diag.std()) if off_diag.size else 0.0

eigvals = np.linalg.eigvalsh(K)
eigvals = np.clip(eigvals, 0, None)

if eigvals.sum() > 0:
    p = eigvals / eigvals.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    vendi_effective_rank = float(np.exp(entropy))
    normalized_vendi = float(vendi_effective_rank / len(texts))
else:
    vendi_effective_rank = 0.0
    normalized_vendi = 0.0

print({
    "n_docs_used": len(texts),
    "embedding_backend": MODEL_NAME,
    "mean_pairwise_cosine_similarity": mean_similarity,
    "std_pairwise_cosine_similarity": std_similarity,
    "vendi_score_effective_rank": vendi_effective_rank,
    "normalized_vendi_score": normalized_vendi,
})