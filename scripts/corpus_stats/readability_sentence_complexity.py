from pathlib import Path
import re
import statistics

try:
    import textstat  # type: ignore[reportMissingImports]
except ImportError:
    textstat = None

INPUT_DIR = Path(r"D:\SL_Medical_Corpus\data\3_agent_export\markdown")

def split_sentences(text: str):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def tokenize(text: str):
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)

sentence_counts = []
avg_sentence_lengths = []
flesch_scores = []
fk_grades = []
fog_scores = []

for fp in sorted(INPUT_DIR.glob("*.md")):
    text = fp.read_text(encoding="utf-8", errors="ignore").strip()
    if len(text) < 200:
        continue

    sentences = split_sentences(text)
    tokens = tokenize(text)

    if not sentences or not tokens:
        continue

    sentence_counts.append(len(sentences))
    avg_sentence_lengths.append(len(tokens) / len(sentences))

    if textstat is not None:
        try:
            flesch_scores.append(textstat.flesch_reading_ease(text))
            fk_grades.append(textstat.flesch_kincaid_grade(text))
            fog_scores.append(textstat.gunning_fog(text))
        except Exception:
            pass

print({
    "documents_used": len(sentence_counts),
    "avg_sentences_per_doc": statistics.mean(sentence_counts) if sentence_counts else None,
    "avg_sentence_length_words": statistics.mean(avg_sentence_lengths) if avg_sentence_lengths else None,
    "flesch_reading_ease_mean": statistics.mean(flesch_scores) if flesch_scores else None,
    "flesch_kincaid_grade_mean": statistics.mean(fk_grades) if fk_grades else None,
    "gunning_fog_mean": statistics.mean(fog_scores) if fog_scores else None,
    "textstat_available": textstat is not None,
})