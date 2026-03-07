#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage A (Multi-turn): manifest.jsonl -> pending_multiturn.jsonl

Creates 2-turn conversations:
system
user  (task + context)
assistant [PENDING]
user  (follow-up)
assistant [PENDING]

- Split by doc_id (no leakage)
- Multilingual + code-mixed prompts
- Uses manifest fields: doc_id, clean_path/raw_path, source_url/source_pdf_url, site, language_primary
"""

import argparse
import os, re, json, random, hashlib
from dataclasses import dataclass
from typing import List, Optional

DEFAULT_SYSTEM = (
   "You are a strict document-grounded assistant. Use ONLY the provided Context. "
    "Do not invent facts. Do not add explanations that are not supported by Context. "
    "Avoid saying 'I don't know' unless the information is truly absent from Context. "
    "If partially answerable, answer the supported part and clearly state what is missing. "
    "If the task asks for a quote, copy text EXACTLY as it appears in Context (verbatim). "
)

TASKS = [
    "extractive_qa",
    "short_answer_qa",
    "summarize",
    "bullet_points",
    "rewrite_patient_friendly",
    "classify_doc_type",
    "mixed_language_qa",
]

DOC_TYPE_LABELS = [
    "guideline", "circular", "notice", "faq", "form", "policy", "press_release",
    "clinical_protocol", "public_health_advice", "unknown"
]

CODEMIX_STYLES = ["plain", "singlish", "tamilish", "mixed"]
MAX_CHUNK_CHARS = 1800


@dataclass
class ManifestRow:
    doc_id: str
    md_path: str
    source_type: str          # "web" | "pdf" | "unknown"
    source_url: str           # for web
    source_pdf_url: str       # for pdf (best)
    source_pdf: str           # pdf filename fallback
    site: str
    language_primary: str


def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def read_jsonl_tolerant(path: str):
    # skips malformed lines
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_ws(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def resolve_path(p: str) -> Optional[str]:
    if not p:
        return None
    p = p.replace("\\", "/")
    if os.path.exists(p):
        return p
    alt = os.path.join(os.getcwd(), p)
    if os.path.exists(alt):
        return alt
    return None


def load_markdown(md_path: str) -> Optional[str]:
    rp = resolve_path(md_path)
    if not rp:
        return None
    with open(rp, "r", encoding="utf-8") as f:
        return normalize_ws(f.read())


def split_into_sections(md: str) -> List[str]:
    parts = re.split(r"\n(?=#{1,6}\s)", md)
    chunks = [normalize_ws(p) for p in parts if normalize_ws(p)]
    if not chunks:
        return [md[:4000]]

    out = []
    for c in chunks:
        if len(c) <= 4000:
            out.append(c)
        else:
            paras = c.split("\n\n")
            buf = ""
            for p in paras:
                if len(buf) + len(p) + 2 > 3800:
                    if buf.strip():
                        out.append(normalize_ws(buf))
                    buf = p
                else:
                    buf = (buf + "\n\n" + p) if buf else p
            if buf.strip():
                out.append(normalize_ws(buf))
    return out[:20] if out else [md[:4000]]


def chunk_is_low_signal(chunk: str) -> bool:
    if len(chunk) < 250:
        return True
    non_heading = re.sub(r"(?m)^#{1,6}\s+.*$", "", chunk).strip()
    if len(non_heading) < 120:
        return True
    return False


def choose_target_lang(main_lang: str, rng: random.Random) -> str:
    langs = ["en", "si", "ta"]
    weights = {"en": 0.34, "si": 0.33, "ta": 0.33}
    if main_lang in weights:
        weights[main_lang] += 0.25
    total = sum(weights.values())
    r = rng.random() * total
    acc = 0.0
    for l in langs:
        acc += weights[l]
        if r <= acc:
            return l
    return main_lang if main_lang in langs else "en"


def codemixify(text: str, style: str) -> str:
    if style == "plain":
        return text
    if style == "singlish":
        prefix, suffix = "Machan, ", " ok da?"
    elif style == "tamilish":
        prefix, suffix = "Anna, ", " seriyaa?"
    else:
        prefix, suffix = "Eh, ", " plz."
    return f"{prefix}{text}{suffix}"


def lang_header(lang: str) -> str:
    if lang == "si":
        return "සිංහලෙන් පිළිතුරු දෙන්න. "
    if lang == "ta":
        return "தமிழில் பதிலளிக்கவும். "
    return "Answer in English. "


def pick_seed_from_chunk(chunk: str, rng: random.Random) -> str:
    sents = re.split(r"(?<=[\.\?\!\u0964\u0DF4\u0D94])\s+|\n", chunk)
    sents = [normalize_ws(s) for s in sents if len(normalize_ws(s)) > 40]
    if not sents:
        return "the document"
    s = rng.choice(sents)
    words = s.split()
    return " ".join(words[: min(10, len(words))])


def build_user_turn_1(task: str, lang: str, seed: str, style: str) -> str:
    if task == "extractive_qa":
        base = f"From the document, answer using an exact quote copied from the context. Topic: {seed}"
    elif task == "short_answer_qa":
        base = f"Answer briefly based only on the document. Topic: {seed}"
    elif task == "summarize":
        base = "Summarize the document section in 3-5 sentences."
    elif task == "bullet_points":
        base = "Extract the key points as 5-8 bullet points."
    elif task == "rewrite_patient_friendly":
        base = "Rewrite this content in patient-friendly language (simple words), without adding new facts."
    elif task == "classify_doc_type":
        base = f"Classify the document type into one of: {', '.join(DOC_TYPE_LABELS)}. Reply with only the label."
    elif task == "mixed_language_qa":
        base = f"Answer based only on the context. Keep it clear. Topic: {seed}"
    else:
        base = f"Use the document to complete the task: {task}"
    return codemixify(lang_header(lang) + base, style)


def build_user_turn_2_followup(task: str, lang: str, style: str) -> str:
    # Universal: prefer short exact quotes, not "exact line(s)"
    if task in ("short_answer_qa", "mixed_language_qa"):
        base = (
            "Follow-up: Quote ONE short exact phrase from the text that directly supports your answer. "
            "Reply with only the exact quote."
        )

    # For extractive_qa, turn 1 is already an exact quote. Don't ask for lines again.
    elif task == "extractive_qa":
        base = (
            "Follow-up: Briefly explain (in 1 short sentence) what the quoted text means, "
            "without adding any new facts."
        )

    elif task == "summarize":
        base = (
            "Follow-up: List 3 key takeaways as bullet points, based only on the same text."
        )

    elif task == "bullet_points":
        base = (
            "Follow-up: Choose ONE bullet point and rewrite it in a clearer way (1-2 sentences), "
            "using only the text (no new facts)."
        )

    elif task == "rewrite_patient_friendly":
        base = (
            "Follow-up: Rewrite again even simpler (like for a 12-year-old), without adding facts."
        )

    elif task == "classify_doc_type":
        base = (
            "Follow-up: Quote ONE short exact phrase from the text that best supports your label. "
            "If your label is 'unknown', quote a phrase that best describes what the document is "
            "(for example, the page title). Reply with only the exact quote."
        )

    else:
        base = "Follow-up: Add one more detail from the same text (no new facts)."

    return codemixify(lang_header(lang) + base, style)


def best_source_line(doc: ManifestRow) -> str:
    """
    Ensures PDFs don't produce blank Source lines.
    """
    if doc.source_type == "web" and doc.source_url:
        return doc.source_url
    if doc.source_type == "pdf":
        if doc.source_pdf_url:
            return doc.source_pdf_url
        if doc.source_pdf:
            return f"(PDF file) {doc.source_pdf}"
    # fallback
    return doc.source_url or doc.source_pdf_url or ""


def make_example(doc: ManifestRow, chunk: str, task: str, lang: str, style: str, seed: str, split: str) -> dict:
    src = best_source_line(doc)
    user1 = (
        f"{build_user_turn_1(task, lang, seed, style)}\n\n"
        f"Context:\n{chunk}\n\n"
        f"Source: {src}\nSite: {doc.site}\nDocID: {doc.doc_id}"
    )
    user2 = build_user_turn_2_followup(task, lang, style)

    ex_id = stable_hash(doc.doc_id + task + lang + style + seed + chunk[:80])
    return {
        "id": ex_id,
        "doc_id": doc.doc_id,
        "lang": lang,
        "task": task,
        "style": style,
        "split": split,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": user1},
            {"role": "assistant", "content": "[PENDING]"},
            {"role": "user", "content": user2},
            {"role": "assistant", "content": "[PENDING]"},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/2_processed/manifest.jsonl")
    ap.add_argument("--out_path", default="data/4_instruction/pending_multiturn.jsonl")
    ap.add_argument("--eval_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260224)
    ap.add_argument("--max_docs", type=int, default=0, help="0=all")
    ap.add_argument("--per_doc", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    docs: List[ManifestRow] = []
    for row in read_jsonl_tolerant(args.manifest):
        doc_id = str(row.get("doc_id") or row.get("id") or "").strip()
        md_path = (row.get("clean_path") or row.get("raw_path") or "").strip()

        source_type = (row.get("source_type") or "unknown").strip().lower()
        source_url = str(row.get("source_url") or row.get("url") or "").strip()
        source_pdf_url = str(row.get("source_pdf_url") or "").strip()
        source_pdf = str(row.get("source_pdf") or "").strip()

        site = (row.get("site") or row.get("domain") or "").strip()
        language_primary = (row.get("language_primary") or row.get("language") or "en").strip().lower()

        if not doc_id or not md_path:
            continue

        docs.append(ManifestRow(
            doc_id=doc_id,
            md_path=md_path,
            source_type=source_type,
            source_url=source_url,
            source_pdf_url=source_pdf_url,
            source_pdf=source_pdf,
            site=site,
            language_primary=language_primary
        ))

    if args.max_docs and args.max_docs > 0:
        docs = docs[: args.max_docs]

    doc_ids = sorted({d.doc_id for d in docs})
    rng.shuffle(doc_ids)
    n_eval = max(1, int(len(doc_ids) * args.eval_ratio)) if doc_ids else 0
    eval_set = set(doc_ids[:n_eval])

    pending = []
    used_docs = 0
    skipped = 0

    for d in docs:
        md = load_markdown(d.md_path)
        if not md or len(md) < 200:
            skipped += 1
            continue
        used_docs += 1

        chunks = split_into_sections(md)
        split = "eval" if d.doc_id in eval_set else "train"
        main_lang = d.language_primary if d.language_primary in ("en", "si", "ta") else "en"

        made = 0
        for _ in range(args.per_doc * 8):
            if made >= args.per_doc:
                break

            chunk = rng.choice(chunks)[:MAX_CHUNK_CHARS]
            if chunk_is_low_signal(chunk):
                continue

            task = rng.choice(TASKS)
            style = rng.choice(CODEMIX_STYLES)
            lang = choose_target_lang(main_lang, rng)
            seed = pick_seed_from_chunk(chunk, rng)

            pending.append(make_example(d, chunk, task, lang, style, seed, split))
            made += 1

    write_jsonl(args.out_path, pending)
    print(f"Docs parsed: {len(docs)} | Docs used: {used_docs} | Docs skipped (missing/short md): {skipped}")
    print(f"Pending multi-turn examples: {len(pending)}")
    print(f"Wrote: {args.out_path}")


if __name__ == "__main__":
    main()