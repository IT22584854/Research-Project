import json
import re
from pathlib import Path

MANIFEST = Path("data/2_processed/manifest.jsonl")
OUT_DIR = Path("data/3_agent_export/markdown")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-_.]+", "_", name)
    return name[:180] if len(name) > 180 else name

def clear_old_exports() -> int:
    removed = 0
    for old_file in OUT_DIR.glob("*.md"):
        old_file.unlink()
        removed += 1
    return removed

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    removed = clear_old_exports()

    n = 0
    seen_filenames = set()

    with MANIFEST.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            doc_id = rec.get("doc_id")
            if not doc_id:
                raise ValueError(f"Missing doc_id on manifest line {line_no}")

            site = rec.get("site") or "UNKNOWN"
            source_url = rec.get("source_url")
            source_pdf = rec.get("source_pdf")
            title = (rec.get("title") or "").strip()

            header_lines = [
                f"Doc ID: {doc_id}",
                f"Site: {site}",
            ]
            if source_url:
                header_lines.append(f"Source URL: {source_url}")
            if source_pdf:
                header_lines.append(f"Source PDF: {source_pdf}")
            if title:
                header_lines.append(f"Title: {title}")

            header = "\n".join(header_lines).strip() + "\n\n"

            body = rec.get("markdown") or ""

            # Remove duplicate provenance lines only at the top of the body
            body = re.sub(
                r"^(Source URL:\s*.*\n|Source PDF:\s*.*\n|Source PDF URL:\s*.*\n)+\n*",
                "",
                body,
                flags=re.IGNORECASE,
            )

            out_text = header + body.strip() + "\n"

            fname = safe_filename(f"{site}__{doc_id[:16]}.md")

            if fname in seen_filenames:
                raise ValueError(f"Filename collision detected: {fname}")

            seen_filenames.add(fname)

            out_path = OUT_DIR / fname
            out_path.write_text(out_text, encoding="utf-8")
            n += 1

    print(f"Removed old markdown files: {removed}")
    print(f"Wrote {n} markdown files to: {OUT_DIR}")

if __name__ == "__main__":
    main()