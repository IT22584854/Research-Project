from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_DIR = Path(r"D:\SL_Medical_Corpus\data\3_agent_export\markdown")
THRESHOLD = 0.95

doc_ids = []
texts = []

for fp in sorted(INPUT_DIR.glob("*.md")):
    text = fp.read_text(encoding="utf-8", errors="ignore").strip()
    if len(text) < 50:
        continue
    doc_ids.append(fp.stem)
    texts.append(text)

vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=2,
    max_features=50000
)

X = vectorizer.fit_transform(texts)
sim = cosine_similarity(X)

pairs = []
docs_in_near_dups = set()
n_docs = len(texts)

for i in range(n_docs):
    for j in range(i + 1, n_docs):
        if sim[i, j] >= THRESHOLD:
            pairs.append({
                "doc_id_1": doc_ids[i],
                "doc_id_2": doc_ids[j],
                "similarity": float(sim[i, j]),
            })
            docs_in_near_dups.add(doc_ids[i])
            docs_in_near_dups.add(doc_ids[j])

near_duplicate_rate = len(docs_in_near_dups) / n_docs if n_docs else 0.0

print({
    "n_docs": n_docs,
    "near_duplicate_threshold": THRESHOLD,
    "near_duplicate_pairs": len(pairs),
    "docs_in_near_duplicates": len(docs_in_near_dups),
    "near_duplicate_rate": near_duplicate_rate,
})

# Optional: save example pairs
import json
from pathlib import Path
out_path = INPUT_DIR.parent / "near_duplicate_pairs.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for row in pairs:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Saved pairs to:", out_path)