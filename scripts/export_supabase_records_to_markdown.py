import json
import re
from pathlib import Path

MANIFEST = Path("data/2_processed/manifest.jsonl")
OUT_DIR = Path("data/3_agent_export/markdown")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-_.]+", "_", name)
    return name[:180] if len(name) > 180 else name

def main():
    n = 0
    with MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            doc_id = rec["doc_id"]
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

            # ✅ REMOVE duplicate Source URL/PDF line at the very top of the body
            body = re.sub(r"^Source (URL|PDF):\s*.*\n\n?", "", body, flags=re.IGNORECASE)

            out_text = header + body.strip() + "\n"

            fname = safe_filename(f"{site}__{doc_id[:16]}.md")
            out_path = OUT_DIR / fname
            out_path.write_text(out_text, encoding="utf-8")

            n += 1

    print(f"✅ Wrote {n} markdown files to: {OUT_DIR}")

if __name__ == "__main__":
    main()