import streamlit as st
import subprocess
import sys
import json
import csv
import ast
from pathlib import Path
from typing import List
import statistics
import pandas as pd
import re
from datetime import datetime

# =========================
# CONFIGURATION & STYLE
# =========================
st.set_page_config(
    page_title="SL Medical Corpus Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
div.stButton > button {
    background-color: #0F766E;
    color: white;
    border: 1px solid #0B5E57;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1rem;
}
div.stButton > button:hover {
    background-color: #115E59;
    color: white;
    border-color: #134E4A;
}
</style>
""", unsafe_allow_html=True)

# Paths
# This file lives in <repo>/scripts, so repo root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "0_raw"
CLEAN_DIR = DATA_DIR / "1_cleaned"
PROCESSED_DIR = DATA_DIR / "2_processed"
AGENT_EXPORT_DIR = DATA_DIR / "3_agent_export"
EXPORT_MD_DIR = AGENT_EXPORT_DIR / "markdown"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)

# Ensure directories exist
for d in [RAW_DIR, CLEAN_DIR, PROCESSED_DIR, AGENT_EXPORT_DIR, EXPORT_MD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SCRIPTS = {
    "Scraper": PROJECT_ROOT / "scripts" / "scraper.py",
    "PDF2MD": PROJECT_ROOT / "scripts" / "pdfs_to_md.py",
    "FindDup": PROJECT_ROOT / "scripts" / "find_duplicates.py",
    "Dedupe": PROJECT_ROOT / "scripts" / "dedupe_exact_duplicates.py",
    "Scrubber": PROJECT_ROOT / "scripts" / "scrubber.py",
    "Manifest": PROJECT_ROOT / "scripts" / "manifest_generator.py",
    "Upload": PROJECT_ROOT / "scripts" / "upload_manifest_to_supabase.py",
    "Export": PROJECT_ROOT / "scripts" / "export_supabase_records_to_markdown.py"
}

METRIC_SCRIPTS = {
    "Lexical": PROJECT_ROOT / "scripts" / "corpus_stats" / "lexical_diversity.py",
    "NearDup": PROJECT_ROOT / "scripts" / "corpus_stats" / "near_duplicate_rate.py",
    "Readability": PROJECT_ROOT / "scripts" / "corpus_stats" / "readability_sentence_complexity.py",
    "Semantic": PROJECT_ROOT / "scripts" / "corpus_stats" / "semantic_diversity.py",
}

NEAR_DUP_PAIRS_PATH = EXPORT_MD_DIR.parent / "near_duplicate_pairs.jsonl"
LIVE_METRICS_DIR = PROJECT_ROOT / "data" / "8_metrics" / "corpus_stats_live"
LIVE_METRICS_JSON = LIVE_METRICS_DIR / "metrics.json"
LIVE_METRICS_MD = LIVE_METRICS_DIR / "metrics.md"

PRECOMPUTED_METRIC_JSON = [
    LIVE_METRICS_JSON,
    PROJECT_ROOT / "data" / "8_metrics" / "corpus_stats" / "metrics.json",
]

PRECOMPUTED_METRIC_MD = [
    LIVE_METRICS_MD,
    PROJECT_ROOT / "data" / "8_metrics" / "corpus_stats" / "metrics.md",
]

METRIC_SECTION_TITLES = {
    "lexical_diversity": "Lexical Diversity",
    "near_duplicate_rate": "Near Duplicate Rate",
    "readability_sentence_complexity": "Readability and Sentence Complexity",
    "semantic_diversity": "Semantic Diversity",
}

INSTRUCTION_METRICS_ROOT = PROJECT_ROOT / "eval_reports"
INSTRUCTION_METRICS_SETS = {
    "Training Set (research_metrics)": INSTRUCTION_METRICS_ROOT / "research_metrics",
    "Eval Set (eval_metrics)": INSTRUCTION_METRICS_ROOT / "eval_metrics",
    "Gold Eval Set (gold_eval_metrics_final)": INSTRUCTION_METRICS_ROOT / "gold_eval_metrics_final",
}

# =========================
# CORE UTILITIES
# =========================
def run_pipeline_step(step_name: str, cmd: List[str]):
    with st.status(f"Executing: {step_name}", expanded=True) as status:
        if not PYTHON_EXE.exists():
            st.error(f"Python interpreter not found: {PYTHON_EXE}")
            status.update(label=f" {step_name} Failed", state="error")
            return None

        process = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, encoding="utf-8", errors="replace")

        if process.stdout.strip():
            st.text_area(f"{step_name} output", process.stdout[-4000:], height=200)
        
        if process.returncode == 0:
            status.update(label=f" {step_name} Complete", state="complete", expanded=False)
        else:
            st.error(process.stderr[-2000:])
            status.update(label=f" {step_name} Failed", state="error")
        return process

def get_count(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.rglob(pattern) if _.is_file())


def get_count_with_history(path: Path, patterns: List[str]):
    total = 0
    history = 0
    for pattern in patterns:
        for fp in path.rglob(pattern):
            if not fp.is_file():
                continue
            total += 1
            if "history" in {part.lower() for part in fp.parts}:
                history += 1
    return total, history

def read_text_safe(path: Path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except:
        return "File not found or unreadable."


def load_json_safe(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _get_nested(payload: dict, *keys):
    cur = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _format_kpi(value, fmt: str = "number"):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if fmt == "percent" and isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    if fmt == "float" and isinstance(value, (int, float)):
        return f"{value:.4f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _distribution_df(payload: dict, section_key: str, label_col: str):
    section = payload.get(section_key)
    if not isinstance(section, dict) or not section:
        return None
    rows = [{label_col: k, "count": v} for k, v in section.items()]
    df = pd.DataFrame(rows)
    if "count" in df.columns:
        df = df.sort_values("count", ascending=False)
    return df


def render_corpus_metrics_dashboard(payload: dict):
    if not isinstance(payload, dict):
        st.error("Corpus metrics payload is not a valid JSON object.")
        return

    sections = payload.get("sections")
    if not isinstance(sections, dict) or not sections:
        st.warning("This metrics file does not match the corpus metrics schema (missing `sections`).")
        return

    lexical = sections.get("lexical_diversity", {}) if isinstance(sections.get("lexical_diversity"), dict) else {}
    near_dup = sections.get("near_duplicate_rate", {}) if isinstance(sections.get("near_duplicate_rate"), dict) else {}
    readability = sections.get("readability_sentence_complexity", {}) if isinstance(sections.get("readability_sentence_complexity"), dict) else {}
    semantic = sections.get("semantic_diversity", {}) if isinstance(sections.get("semantic_diversity"), dict) else {}

    st.markdown("### KPI Overview")
    r1 = st.columns(4)
    r1[0].metric("Documents", _format_kpi(lexical.get("documents_used")))
    r1[1].metric("Total Tokens", _format_kpi(lexical.get("total_tokens")))
    r1[2].metric("Vocabulary", _format_kpi(lexical.get("vocab_size")))
    r1[3].metric("Type-Token Ratio", _format_kpi(lexical.get("type_token_ratio"), "float"))

    r2 = st.columns(4)
    r2[0].metric("Near-Dup Rate", _format_kpi(near_dup.get("near_duplicate_rate"), "percent"))
    r2[1].metric("Near-Dup Pairs", _format_kpi(near_dup.get("near_duplicate_pairs")))
    r2[2].metric("Avg Sentence Length", _format_kpi(readability.get("avg_sentence_length_words"), "float"))
    r2[3].metric("Mean Similarity", _format_kpi(semantic.get("mean_similarity"), "float"))

    st.markdown("### Graphs")
    snapshot_rows = [
        {"metric": "type_token_ratio", "value": lexical.get("type_token_ratio")},
        {"metric": "near_duplicate_rate", "value": near_dup.get("near_duplicate_rate")},
        {"metric": "avg_sentence_length_words", "value": readability.get("avg_sentence_length_words")},
        {"metric": "mean_similarity", "value": semantic.get("mean_similarity")},
    ]
    snapshot_df = pd.DataFrame(snapshot_rows)
    snapshot_df = snapshot_df[pd.to_numeric(snapshot_df["value"], errors="coerce").notna()]
    if not snapshot_df.empty:
        snapshot_df["value"] = pd.to_numeric(snapshot_df["value"], errors="coerce")
        st.bar_chart(snapshot_df.set_index("metric"))

    gc1, gc2 = st.columns(2)

    with gc1:
        st.markdown("**Lexical Diversity**")
        lex_numeric = {
            k: v for k, v in lexical.items()
            if isinstance(v, (int, float)) and k not in {"total_tokens", "vocab_size", "documents_used"}
        }
        if lex_numeric:
            lex_df = pd.DataFrame([{"metric": k, "value": v} for k, v in lex_numeric.items()])
            st.bar_chart(lex_df.set_index("metric"))
        else:
            st.info("No numeric lexical metrics available for charting.")

    with gc2:
        st.markdown("**Readability and Sentence Complexity**")
        read_numeric = {k: v for k, v in readability.items() if isinstance(v, (int, float))}
        if read_numeric:
            read_df = pd.DataFrame([{"metric": k, "value": v} for k, v in read_numeric.items()])
            st.bar_chart(read_df.set_index("metric"))
        else:
            st.info("No numeric readability metrics available for charting.")

    gc3, gc4 = st.columns(2)
    with gc3:
        st.markdown("**Near Duplicate Rate**")
        near_numeric = {k: v for k, v in near_dup.items() if isinstance(v, (int, float))}
        if near_numeric:
            near_df = pd.DataFrame([{"metric": k, "value": v} for k, v in near_numeric.items()])
            st.bar_chart(near_df.set_index("metric"))
        else:
            st.info("No numeric near-duplicate metrics available for charting.")

    with gc4:
        st.markdown("**Semantic Diversity**")
        sem_numeric = {k: v for k, v in semantic.items() if isinstance(v, (int, float))}
        if sem_numeric:
            sem_df = pd.DataFrame([{"metric": k, "value": v} for k, v in sem_numeric.items()])
            st.bar_chart(sem_df.set_index("metric"))
        else:
            st.info("No numeric semantic metrics available for charting.")

    st.markdown("### Section Details")
    for section_key, values in sections.items():
        if not isinstance(values, dict):
            continue
        title = METRIC_SECTION_TITLES.get(section_key, section_key).title()
        with st.expander(title, expanded=(section_key == "lexical_diversity")):
            df = pd.DataFrame([{"metric": k, "value": v} for k, v in values.items()])
            st.dataframe(df, use_container_width=True, hide_index=True)


def render_metrics_dashboard(payload: dict):
    if not isinstance(payload, dict):
        st.error("Metrics payload is not a valid JSON object.")
        return

    kpis = [
        ("Final Records", _get_nested(payload, "dataset_cleaning", "final_total_records"), "number"),
        ("Retention Rate", _get_nested(payload, "dataset_cleaning", "retention_rate"), "percent"),
        ("Duplicate Rate", _get_nested(payload, "duplicates", "duplicate_rate"), "percent"),
        ("Over-Refusal Rate", _get_nested(payload, "refusal_metrics", "over_refusal_rate"), "percent"),
        ("Linguistic Diversity", _get_nested(payload, "distribution_metrics", "linguistic_diversity_index"), "float"),
        ("Type-Token Ratio", _get_nested(payload, "ttr", "type_token_ratio"), "float"),
        ("Normalized Vendi", _get_nested(payload, "vendi", "normalized_vendi_score"), "float"),
        ("Context Similarity", _get_nested(payload, "context_answer_similarity", "mean_similarity"), "float"),
    ]

    st.markdown("### KPI Overview")
    row1 = st.columns(4)
    for idx, (label, value, fmt) in enumerate(kpis[:4]):
        row1[idx].metric(label, _format_kpi(value, fmt))

    row2 = st.columns(4)
    for idx, (label, value, fmt) in enumerate(kpis[4:]):
        row2[idx].metric(label, _format_kpi(value, fmt))

    st.markdown("### Distributions")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Task Distribution**")
        task_df = _distribution_df(payload, "task_distribution", "task")
        if task_df is not None:
            st.bar_chart(task_df.set_index("task"))
            st.dataframe(task_df, use_container_width=True, hide_index=True)
        else:
            st.info("Task distribution is not available.")

    with d2:
        st.markdown("**Language Distribution**")
        lang_df = _distribution_df(payload, "language_distribution", "language")
        if lang_df is not None:
            st.bar_chart(lang_df.set_index("language"))
            st.dataframe(lang_df, use_container_width=True, hide_index=True)
        else:
            st.info("Language distribution is not available.")

    st.markdown("### Detailed Sections")
    skip_sections = {"task_distribution", "language_distribution"}
    for section_name, section_values in payload.items():
        if section_name in skip_sections:
            continue
        if not isinstance(section_values, dict):
            continue
        with st.expander(section_name.replace("_", " ").title(), expanded=False):
            section_df = pd.DataFrame(
                [{"metric": k, "value": v} for k, v in section_values.items()]
            )
            st.dataframe(section_df, use_container_width=True, hide_index=True)


def read_jsonl_rows(path: Path, limit: int = 200):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def parse_metrics_dict_from_stdout(stdout_text: str):
    """Parse the first dict-like line printed by corpus_stats scripts."""
    for line in (stdout_text or "").splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def _fmt_value(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_live_metrics_reports(payload: dict):
    LIVE_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_METRICS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Live Corpus Stats Metrics",
        "",
        f"Generated at: {payload.get('generated_at', '')}",
        "",
    ]

    sections = payload.get("sections", {})
    for key, section_obj in sections.items():
        lines.append(f"## {METRIC_SECTION_TITLES.get(key, key)}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for metric_name, metric_value in section_obj.items():
            lines.append(f"| {metric_name} | {_fmt_value(metric_value)} |")
        lines.append("")

    LIVE_METRICS_MD.write_text("\n".join(lines), encoding="utf-8")


def update_live_metrics_section(section_key: str, section_metrics: dict):
    payload = load_json_safe(LIVE_METRICS_JSON)
    if not isinstance(payload, dict):
        payload = {"generated_at": "", "sections": {}}

    sections = payload.get("sections")
    if not isinstance(sections, dict):
        sections = {}

    sections[section_key] = section_metrics
    payload["sections"] = sections
    payload["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    write_live_metrics_reports(payload)


def run_metric_script_and_update(step_name: str, script_path: Path, section_key: str):
    process = run_pipeline_step(step_name, [str(PYTHON_EXE), str(script_path)])
    if process is None or process.returncode != 0:
        return

    parsed = parse_metrics_dict_from_stdout(process.stdout or "")
    if not parsed:
        st.warning(f"{step_name} ran, but no parseable metrics dictionary was found in stdout.")
        return

    update_live_metrics_section(section_key, parsed)
    st.success(
        f"Updated live metrics reports: `{LIVE_METRICS_JSON.relative_to(PROJECT_ROOT)}` and `{LIVE_METRICS_MD.relative_to(PROJECT_ROOT)}`"
    )

# =========================
# METRICS HELPERS
# =========================
def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)

def split_sentences(text: str):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def moving_average_ttr(tokens, window_size=500):
    if not tokens: return 0.0
    if len(tokens) < window_size: return len(set(tokens)) / len(tokens)
    vals = [len(set(tokens[i:i + window_size])) / window_size for i in range(len(tokens) - window_size + 1)]
    return sum(vals) / len(vals)

def compute_lexical_metrics(input_dir: Path):
    all_tokens, doc_token_counts = [], []
    files = sorted(list(input_dir.glob("*.md")))
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        toks = tokenize(text)
        if toks:
            all_tokens.extend(toks)
            doc_token_counts.append(len(toks))
    if not all_tokens: return None
    return {
        "documents_used": len(doc_token_counts),
        "total_tokens": len(all_tokens),
        "vocab_size": len(set(all_tokens)),
        "type_token_ratio": len(set(all_tokens)) / len(all_tokens),
        "mattr_500": moving_average_ttr(all_tokens, 500),
        "avg_tokens_per_doc": statistics.mean(doc_token_counts)
    }

def compute_near_duplicate_metrics(input_dir: Path, threshold: float = 0.95):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    doc_ids, texts = [], []
    for fp in sorted(input_dir.glob("*.md")):
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) > 50:
            doc_ids.append(fp.stem)
            texts.append(text)
    if not texts: return {"n_docs": 0}, []
    X = TfidfVectorizer(lowercase=True, min_df=2, max_features=10000).fit_transform(texts)
    sim = cosine_similarity(X)
    pairs, docs_in_near_dups = [], set()
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if sim[i, j] >= threshold:
                pairs.append({"doc1": doc_ids[i], "doc2": doc_ids[j], "similarity": float(sim[i, j])})
                docs_in_near_dups.add(doc_ids[i]); docs_in_near_dups.add(doc_ids[j])
    return {
        "n_docs": len(texts), "near_duplicate_threshold": threshold,
        "near_duplicate_pairs": len(pairs), "docs_in_near_duplicates": len(docs_in_near_dups),
        "near_duplicate_rate": len(docs_in_near_dups) / len(texts) if texts else 0
    }, pairs

def compute_readability_metrics(input_dir: Path):
    try: import textstat
    except: textstat = None
    sentence_counts, avg_sentence_lengths, flesch, fk = [], [], [], []
    for fp in sorted(input_dir.glob("*.md")):
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) < 100: continue
        sents = split_sentences(text); toks = tokenize(text)
        if sents and toks:
            sentence_counts.append(len(sents))
            avg_sentence_lengths.append(len(toks)/len(sents))
            if textstat:
                flesch.append(textstat.flesch_reading_ease(text))
                fk.append(textstat.flesch_kincaid_grade(text))
    return {
        "documents_used": len(sentence_counts),
        "avg_sentences_per_doc": statistics.mean(sentence_counts) if sentence_counts else 0,
        "avg_sentence_length_words": statistics.mean(avg_sentence_lengths) if avg_sentence_lengths else 0,
        "flesch_reading_ease_mean": statistics.mean(flesch) if flesch else None,
        "flesch_kincaid_grade_mean": statistics.mean(fk) if fk else None,
        "textstat_available": textstat is not None
    }

def compute_embedding_diversity_metrics(input_dir: Path, model_name: str, max_docs: int):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    texts = [fp.read_text(encoding="utf-8", errors="ignore")[:4000] for fp in sorted(input_dir.glob("*.md")) if fp.stat().st_size > 100]
    if len(texts) > max_docs: texts = texts[:max_docs]
    if not texts: return {"n_docs_used": 0}
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True)
    K = emb @ emb.T
    off_diag = K[~np.eye(K.shape[0], dtype=bool)]
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.clip(eigvals, 0, None)
    p = eigvals / eigvals.sum()
    vendi = np.exp(-np.sum(p[p>0] * np.log(p[p>0])))
    return {
        "n_docs_used": len(texts), "embedding_backend": model_name,
        "mean_similarity": float(off_diag.mean()), "vendi_score": float(vendi)
    }

# =========================
# MAIN UI
# =========================
st.title(" Sri Lankan Medical Corpus Pipeline")
st.caption("Automated processing for clinical data ingestion and RAG preparation.")
st.sidebar.caption(f"Python: `{PYTHON_EXE}`")

st.subheader("Script Verification")
for name, path in SCRIPTS.items():
    icon = "" if path.exists() else ""
    st.caption(f"{icon} **{name}** - `{path.relative_to(PROJECT_ROOT)}`")

# Stats Header
raw_total, raw_history = get_count_with_history(RAW_DIR, ["*.md", "*.pdf"])
clean_total, clean_history = get_count_with_history(CLEAN_DIR, ["*.md"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw Assets", raw_total, delta=f"history: {raw_history}")
c2.metric("Cleaned Docs", clean_total, delta=f"history: {clean_history}")
c3.metric("Manifests", get_count(PROCESSED_DIR, "manifest.jsonl"))
c4.metric("Agent Exports", get_count(AGENT_EXPORT_DIR, "*.md"))

st.divider()

tab_names = ["Ingest", "Process", "Clean", "Manifest", "Cloud", "Export", "Metrics", "Instruction Metrics"]
active_tab = st.sidebar.radio("Navigate", tab_names, index=0)

if active_tab == "Ingest":
    st.subheader(" Data Acquisition")
    col1, col2 = st.columns([1, 2])
    with col1:
        bf = st.checkbox("Backfill Mode", value=True)
        conc = st.number_input("Threads", 1, 32, 6)
        if st.button("Run Scraper"):
            cmd = [str(PYTHON_EXE), str(SCRIPTS["Scraper"]), "--concurrency", str(conc)]
            if bf: cmd.append("--backfill")
            run_pipeline_step("Web Scraper", cmd)
    with col2:
        raw_list = sorted(list(RAW_DIR.glob("*.md")), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        if raw_list:
            pick = st.selectbox("Select file:", raw_list, format_func=lambda x: x.name)
            st.text_area("Preview", read_text_safe(pick), height=200)

elif active_tab == "Process":
    st.subheader("️ PDF Transformation")
    if st.button("Convert PDFs to Markdown"):
        run_pipeline_step("PDF Converter", [str(PYTHON_EXE), str(SCRIPTS["PDF2MD"])])

elif active_tab == "Clean":
    st.subheader(" Cleaning & Deduplication")
    k1, k2, k3 = st.columns(3)
    with k1:
        if st.button(" Run Scrubber"): run_pipeline_step("Scrubber", [str(PYTHON_EXE), str(SCRIPTS["Scrubber"])])
    with k2:
        if st.button(" Find Dups"): run_pipeline_step("Duplicate Search", [str(PYTHON_EXE), str(SCRIPTS["FindDup"])])
    with k3:
        if st.button("️ Deduplicate"): run_pipeline_step("Deduplication", [str(PYTHON_EXE), str(SCRIPTS["Dedupe"])])

elif active_tab == "Manifest":
    st.subheader(" Manifest Generator")
    if st.button("Build Final Manifest"):
        run_pipeline_step("Manifest Gen", [str(PYTHON_EXE), str(SCRIPTS["Manifest"])])
    m_path = PROCESSED_DIR / "manifest.jsonl"
    if m_path.exists():
        with open(m_path, "r", encoding="utf-8", errors="ignore") as f:
            preview = [json.loads(line) for _, line in zip(range(3), f)]
            st.json(preview)

elif active_tab == "Cloud":
    st.subheader("️ Supabase Upload")
    if st.button("Sync to Supabase"):
        run_pipeline_step("Cloud Sync", [str(PYTHON_EXE), str(SCRIPTS["Upload"])])

elif active_tab == "Export":
    st.subheader(" Agent Handoff")
    if st.button("Export Agent Markdown"):
        run_pipeline_step("Agent Export", [str(PYTHON_EXE), str(SCRIPTS["Export"])])

elif active_tab == "Metrics":
    st.subheader(" Corpus Quality Analysis")
    st.caption("Run metrics scripts on demand, then view precomputed outputs below.")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        if st.button("Run Lexical Metrics"):
            run_metric_script_and_update(
                "Lexical Diversity",
                METRIC_SCRIPTS["Lexical"],
                section_key="lexical_diversity",
            )
    with r2:
        if st.button("Run Near-Duplicate Metrics"):
            run_metric_script_and_update(
                "Near Duplicate Rate",
                METRIC_SCRIPTS["NearDup"],
                section_key="near_duplicate_rate",
            )
    with r3:
        if st.button("Run Readability Metrics"):
            run_metric_script_and_update(
                "Readability and Sentence Complexity",
                METRIC_SCRIPTS["Readability"],
                section_key="readability_sentence_complexity",
            )
    with r4:
        if st.button("Run Semantic Metrics"):
            run_metric_script_and_update(
                "Semantic Diversity",
                METRIC_SCRIPTS["Semantic"],
                section_key="semantic_diversity",
            )

    available_json = [p for p in PRECOMPUTED_METRIC_JSON if p.exists()]
    available_md = [p for p in PRECOMPUTED_METRIC_MD if p.exists()]

    if not available_json and not available_md:
        st.warning("No precomputed metrics files found under eval_reports/ or data/8_metrics.")
    else:
        view_mode = st.radio("View", ["Summary Metrics", "Markdown Report", "Near-Duplicate Pairs"], horizontal=True)

        if view_mode == "Summary Metrics":
            if not available_json:
                st.info("No metrics.json found.")
            else:
                selected_json = st.selectbox(
                    "Select metrics.json",
                    available_json,
                    format_func=lambda p: str(p.relative_to(PROJECT_ROOT)),
                )
                payload = load_json_safe(selected_json)
                if payload is None:
                    st.error(f"Could not parse JSON: {selected_json}")
                else:
                    render_corpus_metrics_dashboard(payload)
                    with st.expander("Raw metrics.json", expanded=False):
                        st.json(payload)

        elif view_mode == "Markdown Report":
            if not available_md:
                st.info("No metrics.md found.")
            else:
                selected_md = st.selectbox(
                    "Select metrics.md",
                    available_md,
                    format_func=lambda p: str(p.relative_to(PROJECT_ROOT)),
                )
                st.markdown(read_text_safe(selected_md))

        elif view_mode == "Near-Duplicate Pairs":
            if not NEAR_DUP_PAIRS_PATH.exists():
                st.info(f"Pairs file not found: {NEAR_DUP_PAIRS_PATH.relative_to(PROJECT_ROOT)}")
            else:
                max_rows = st.number_input("Rows to preview", min_value=10, max_value=2000, value=200, step=10)
                pairs = read_jsonl_rows(NEAR_DUP_PAIRS_PATH, limit=int(max_rows))
                st.write(f"Previewing {len(pairs)} rows from `{NEAR_DUP_PAIRS_PATH.relative_to(PROJECT_ROOT)}`")
                if pairs:
                    st.dataframe(pd.DataFrame(pairs), use_container_width=True)

elif active_tab == "Instruction Metrics":
    st.subheader(" Instruction Dataset Metrics")
    st.caption("Select a dataset using buttons, then view its JSON and Markdown reports.")

    missing_sets = [name for name, path in INSTRUCTION_METRICS_SETS.items() if not path.exists()]
    if missing_sets:
        st.warning("Missing expected metric folders: " + ", ".join(missing_sets))

    dataset_names = list(INSTRUCTION_METRICS_SETS.keys())
    state_key = "instruction_metrics_selected_dataset"
    if state_key not in st.session_state:
        st.session_state[state_key] = dataset_names[0]

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Training Set", use_container_width=True):
            st.session_state[state_key] = dataset_names[0]
    with b2:
        if st.button("Eval Set", use_container_width=True):
            st.session_state[state_key] = dataset_names[1]
    with b3:
        if st.button("Gold Eval Set", use_container_width=True):
            st.session_state[state_key] = dataset_names[2]

    section_name = st.session_state[state_key]
    section_dir = INSTRUCTION_METRICS_SETS[section_name]
    st.markdown(f"### {section_name}")

    json_path = section_dir / "metrics.json"
    md_path = section_dir / "metrics.md"

    if json_path.exists():
        payload = load_json_safe(json_path)
        if payload is None:
            st.error(f"Could not parse JSON: {json_path.relative_to(PROJECT_ROOT)}")
        else:
            render_metrics_dashboard(payload)
            with st.expander("Raw metrics.json", expanded=False):
                st.json(payload)
    else:
        st.info(f"Not found: `{json_path.relative_to(PROJECT_ROOT)}`")

    with st.expander("Markdown Report", expanded=False):
        if md_path.exists():
            st.markdown(read_text_safe(md_path))
        else:
            st.info(f"Not found: `{md_path.relative_to(PROJECT_ROOT)}`")

    st.divider()

st.divider()
st.caption("Sri Lankan Medical Corpus Project | v2.2")