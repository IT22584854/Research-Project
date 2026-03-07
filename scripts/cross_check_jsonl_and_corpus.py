import os, json
from dotenv import load_dotenv
from supabase import create_client
from collections import Counter

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

train = load_jsonl("data/4_instruction/train_multiturn.jsonl")
eval_  = load_jsonl("data/4_instruction/eval_multiturn.jsonl")

all_doc_ids = sorted(set([r["doc_id"] for r in train] + [r["doc_id"] for r in eval_]))

def fetch_corpus_rows(doc_ids_chunk):
    # Only pull what we need for analysis (faster)
    res = (
        supabase.table("sl_med_corpus")
        .select("doc_id,site,language_primary,source_type,bytes,retrieved_at,clean_path,refined_path,markdown")
        .in_("doc_id", doc_ids_chunk)
        .execute()
    )
    return res.data

found = {}
site_ct = Counter()
lang_ct = Counter()
source_ct = Counter()
markdown_missing = 0

CHUNK = 300
for i in range(0, len(all_doc_ids), CHUNK):
    chunk = all_doc_ids[i:i+CHUNK]
    data = fetch_corpus_rows(chunk)
    for row in data:
        did = row["doc_id"]
        found[did] = row
        if row.get("site"): site_ct[row["site"]] += 1
        if row.get("language_primary"): lang_ct[row["language_primary"]] += 1
        if row.get("source_type"): source_ct[row["source_type"]] += 1
        md = row.get("markdown")
        if md is None or len(md) < 200:
            markdown_missing += 1

missing = [d for d in all_doc_ids if d not in found]

print("=== Dataset ↔ Corpus check ===")
print("Dataset unique doc_ids:", len(all_doc_ids))
print("Found in sl_med_corpus:", len(found))
print("Missing from sl_med_corpus:", len(missing))
if missing:
    print("Example missing doc_ids:", missing[:20])

print("\n=== Coverage of used docs (by metadata) ===")
print("Top sites:", site_ct.most_common(15))
print("Language_primary:", lang_ct.most_common())
print("Source_type:", source_ct.most_common())
print("Used docs with missing/short markdown (<200 chars):", markdown_missing)