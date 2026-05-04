import re
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_XLSX = r"C:\Users\Charunya\Desktop\medical_translation_chunks.csv"
OUTPUT_MERGED_XLSX = r"D:\SL_Medical_Corpus\scripts\merged_translations.csv"
OUTPUT_BOUNDARY_REPORT_XLSX = r"D:\SL_Medical_Corpus\scripts\merge_boundary_report.csv"

KNOWN_OVERLAP_CHARS = 300

# For exact overlap search window
MIN_EXACT_OVERLAP = 80
MAX_EXACT_OVERLAP = 400

# For fuzzy overlap fallback
FUZZY_WINDOW = 450
FUZZY_MIN_MATCH = 60
FUZZY_SIM_THRESHOLD = 0.82


# =========================
# TEXT HELPERS
# =========================
def normalize_for_match(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_preserve_structure(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def starts_like_new_block(text: str) -> bool:
    text = text.lstrip()
    if not text:
        return False

    patterns = [
        r"^#{1,6}\s",            # markdown heading
        r"^\d+[\.\)]\s",         # numbered list
        r"^[\-\*\u2022]\s",      # bullets
        r"^[A-Z][A-Za-z0-9 ,:/\-\(\)]{0,80}$",  # short title-like line
    ]

    return any(re.match(p, text) for p in patterns)


def ends_with_sentence_punct(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text[-1] in ".?!:;।"


# =========================
# OVERLAP DETECTION
# =========================
def find_exact_suffix_prefix_overlap(prev_text: str, next_text: str,
                                     min_overlap: int = MIN_EXACT_OVERLAP,
                                     max_overlap: int = MAX_EXACT_OVERLAP) -> int:
    """
    Find longest exact overlap where suffix of prev_text == prefix of next_text.
    Works best for source_text or when translated overlap stayed nearly identical.
    """
    prev_n = normalize_for_match(prev_text)
    next_n = normalize_for_match(next_text)

    max_len = min(len(prev_n), len(next_n), max_overlap)
    best = 0

    for k in range(max_len, min_overlap - 1, -1):
        if prev_n[-k:] == next_n[:k]:
            best = k
            break

    return best


def find_fuzzy_overlap(prev_text: str, next_text: str,
                       window: int = FUZZY_WINDOW,
                       min_match: int = FUZZY_MIN_MATCH,
                       sim_threshold: float = FUZZY_SIM_THRESHOLD):
    """
    Fuzzy fallback for translated text.
    Compares end of previous chunk and beginning of next chunk.
    Returns estimated overlap length in next_text.
    """
    prev_raw = clean_text_preserve_structure(prev_text)
    next_raw = clean_text_preserve_structure(next_text)

    prev_tail = prev_raw[-window:]
    next_head = next_raw[:window]

    if not prev_tail or not next_head:
        return 0, 0.0, ""

    matcher = SequenceMatcher(None, prev_tail, next_head)
    match = matcher.find_longest_match(0, len(prev_tail), 0, len(next_head))

    if match.size >= min_match:
        overlap_text = next_head[match.b:match.b + match.size]
        similarity = match.size / max(len(overlap_text), 1)
        if similarity >= sim_threshold:
            estimated_overlap = match.b + match.size
            return estimated_overlap, similarity, overlap_text

    return 0, 0.0, ""


# =========================
# MERGE LOGIC
# =========================
def join_without_dup(prev_text: str, next_text: str):
    """
    Merge two chunks.
    Priority:
    1. exact overlap
    2. fuzzy overlap
    3. mid-sentence join
    4. paragraph join
    """
    prev_clean = clean_text_preserve_structure(prev_text)
    next_clean = clean_text_preserve_structure(next_text)

    if not prev_clean:
        return next_clean, {
            "method": "init",
            "overlap_len": 0,
            "overlap_preview": "",
            "needs_review": False
        }

    if not next_clean:
        return prev_clean, {
            "method": "empty_next",
            "overlap_len": 0,
            "overlap_preview": "",
            "needs_review": False
        }

    # 1) exact overlap
    exact_overlap = find_exact_suffix_prefix_overlap(prev_clean, next_clean)
    if exact_overlap > 0:
        merged = prev_clean + next_clean[exact_overlap:]
        return merged, {
            "method": "exact_overlap_trim",
            "overlap_len": exact_overlap,
            "overlap_preview": next_clean[:min(exact_overlap, 120)],
            "needs_review": False
        }

    # 2) fuzzy overlap
    fuzzy_overlap, sim, overlap_text = find_fuzzy_overlap(prev_clean, next_clean)
    if fuzzy_overlap > 0:
        merged = prev_clean.rstrip() + " " + next_clean[fuzzy_overlap:].lstrip()
        return merged, {
            "method": "fuzzy_overlap_trim",
            "overlap_len": fuzzy_overlap,
            "overlap_preview": overlap_text[:120],
            "needs_review": False if sim >= 0.9 else True
        }

    # 3) mid-sentence continuation
    prev_end_sentence = ends_with_sentence_punct(prev_clean)
    next_lstrip = next_clean.lstrip()
    starts_lower = bool(next_lstrip) and next_lstrip[0].islower()
    new_block = starts_like_new_block(next_clean)

    if (not prev_end_sentence and not new_block) or starts_lower:
        merged = prev_clean.rstrip() + " " + next_clean.lstrip()
        return merged, {
            "method": "mid_sentence_join",
            "overlap_len": 0,
            "overlap_preview": "",
            "needs_review": True
        }

    # 4) paragraph join
    merged = prev_clean.rstrip() + "\n\n" + next_clean.lstrip()
    return merged, {
        "method": "paragraph_join",
        "overlap_len": 0,
        "overlap_preview": "",
        "needs_review": False
    }


def merge_group(group: pd.DataFrame):
    group = group.sort_values("chunk_index").copy()

    merged_source = ""
    merged_translation = ""
    boundary_rows = []

    for i, row in group.iterrows():
        chunk_idx = int(row["chunk_index"])
        source_chunk = row.get("source_text", "")
        translated_chunk = row.get("translated_text", "")

        # Merge source
        merged_source, src_meta = join_without_dup(merged_source, source_chunk)

        # Merge translation
        merged_translation, trg_meta = join_without_dup(merged_translation, translated_chunk)

        boundary_rows.append({
            "doc_id": row["doc_id"],
            "target_lang": row["target_lang"],
            "chunk_index": chunk_idx,
            "source_merge_method": src_meta["method"],
            "source_overlap_len": src_meta["overlap_len"],
            "source_overlap_preview": src_meta["overlap_preview"],
            "source_needs_review": src_meta["needs_review"],
            "translation_merge_method": trg_meta["method"],
            "translation_overlap_len": trg_meta["overlap_len"],
            "translation_overlap_preview": trg_meta["overlap_preview"],
            "translation_needs_review": trg_meta["needs_review"],
        })

    merged_row = {
        "doc_id": group["doc_id"].iloc[0],
        "target_lang": group["target_lang"].iloc[0],
        "num_chunks": len(group),
        "chunk_indexes": ", ".join(map(str, group["chunk_index"].tolist())),
        "merged_source_text": merged_source,
        "merged_translated_text": merged_translation,
    }

    return merged_row, boundary_rows


# =========================
# MAIN
# =========================
def main():
    input_path = Path(INPUT_XLSX)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = {
        "doc_id",
        "chunk_index",
        "target_lang",
        "source_text",
        "translated_text",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only successful rows if status exists
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "success"].copy()

    df["chunk_index"] = pd.to_numeric(df["chunk_index"], errors="coerce")
    df = df.dropna(subset=["doc_id", "chunk_index", "target_lang"]).copy()
    df["chunk_index"] = df["chunk_index"].astype(int)

    merged_rows = []
    boundary_report_rows = []

    grouped = df.groupby(["doc_id", "target_lang"], sort=False)

    for (_, _), group in grouped:
        merged_row, boundary_rows = merge_group(group)
        merged_rows.append(merged_row)
        boundary_report_rows.extend(boundary_rows)

    merged_df = pd.DataFrame(merged_rows).sort_values(["doc_id", "target_lang"])
    boundary_df = pd.DataFrame(boundary_report_rows).sort_values(
        ["doc_id", "target_lang", "chunk_index"]
    )

    merged_df.to_csv(OUTPUT_MERGED_XLSX, index=False)
    boundary_df.to_csv(OUTPUT_BOUNDARY_REPORT_XLSX, index=False)

    print(f"Saved merged file: {OUTPUT_MERGED_XLSX}")
    print(f"Saved boundary report: {OUTPUT_BOUNDARY_REPORT_XLSX}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Boundary rows: {len(boundary_df)}")


if __name__ == "__main__":
    main()