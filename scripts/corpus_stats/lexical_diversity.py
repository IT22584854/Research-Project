from pathlib import Path
import re
import statistics

INPUT_DIR = Path(r"D:\SL_Medical_Corpus\data\3_agent_export\markdown")

def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)

def moving_average_ttr(tokens, window_size=500):
    if not tokens:
        return 0.0
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens)
    vals = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        vals.append(len(set(window)) / window_size)
    return sum(vals) / len(vals)

all_tokens = []
doc_token_counts = []

md_files = sorted(INPUT_DIR.glob("*.md"))

for fp in md_files:
    text = fp.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        continue
    toks = tokenize(text)
    if not toks:
        continue
    all_tokens.extend(toks)
    doc_token_counts.append(len(toks))

total_tokens = len(all_tokens)
vocab_size = len(set(all_tokens))
ttr = vocab_size / total_tokens if total_tokens else 0.0
mattr_500 = moving_average_ttr(all_tokens, 500)

print({
    "documents_used": len(doc_token_counts),
    "total_tokens": total_tokens,
    "vocab_size": vocab_size,
    "type_token_ratio": ttr,
    "mattr_500": mattr_500,
    "avg_tokens_per_doc": statistics.mean(doc_token_counts) if doc_token_counts else None,
    "median_tokens_per_doc": statistics.median(doc_token_counts) if doc_token_counts else None,
})