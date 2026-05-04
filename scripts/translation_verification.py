import re
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# ---------- CUDA CHECK ----------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD MODEL ----------
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device=device
)
print("Model device:", model.device)

# ---------- HELPERS ----------
def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def extract_numbers(text):
    return re.findall(r'\d+(?:[.,]\d+)?', clean_text(text))

def extract_percentages(text):
    return re.findall(r'\d+(?:[.,]\d+)?\s*%', clean_text(text))

def extract_negation_like(text):
    text = clean_text(text).lower()
    cues = []
    for word in ["not", "no", "without", "except", "must", "should", "only", "cannot", "never"]:
        if re.search(rf'\b{re.escape(word)}\b', text):
            cues.append(word)
    return cues

# ---------- LOAD DATA ----------
df = pd.read_csv(r"C:\Users\Charunya\Downloads\sl_med_translations_rows (1).csv")

# Clean text columns
df["source_text"] = df["source_text"].apply(clean_text)
df["translation_si"] = df["translation_si"].apply(clean_text)

# ---------- BATCH ENCODE ----------
source_texts = df["source_text"].tolist()
si_texts = df["translation_si"].tolist()

source_emb = model.encode(
    source_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=True
)

si_emb = model.encode(
    si_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=True
)

# Pairwise similarity for matching rows only
similarities = util.cos_sim(source_emb, si_emb).diagonal().cpu().numpy()

# ---------- ANALYZE ----------
rows = []

for i, row in df.iterrows():
    source = row["source_text"]
    si = row["translation_si"]

    sim = float(similarities[i])

    src_numbers = extract_numbers(source)
    si_numbers = extract_numbers(si)
    missing_numbers = [n for n in src_numbers if n not in si_numbers]

    src_perc = extract_percentages(source)
    si_perc = extract_percentages(si)
    missing_percentages = [p for p in src_perc if p not in si_perc]

    src_cues = extract_negation_like(source)

    flags = []

    if sim < 0.65:
        flags.append("low_semantic_similarity")
    if sim < 0.55:
        flags.append("possible_meaning_loss")
    if missing_numbers:
        flags.append("missing_numbers")
    if missing_percentages:
        flags.append("missing_percentages")
    if len(source.split()) >= 20 and sim < 0.70:
        flags.append("review_long_chunk")

    rows.append({
        "semantic_similarity": round(sim, 4),
        "source_numbers": ", ".join(src_numbers),
        "si_numbers": ", ".join(si_numbers),
        "missing_numbers": ", ".join(missing_numbers),
        "missing_percentages": ", ".join(missing_percentages),
        "source_cues": ", ".join(src_cues),
        "flags": ", ".join(flags),
        "flag_count": len(flags),
    })

results = pd.DataFrame(rows)
out = pd.concat([df, results], axis=1)

suspicious = out[out["flag_count"] > 0].copy()
suspicious = suspicious.sort_values(
    by=["flag_count", "semantic_similarity"],
    ascending=[False, True]
)

out.to_csv("translation_meaning_check_all.csv", index=False, encoding="utf-8-sig")
suspicious.to_csv("translation_meaning_check_flagged.csv", index=False, encoding="utf-8-sig")

print("Total rows:", len(out))
print("Flagged rows:", len(suspicious))
print(suspicious[
    ["source_text", "translation_si", "semantic_similarity", "missing_numbers", "missing_percentages", "flags"]
].head(20).to_string())