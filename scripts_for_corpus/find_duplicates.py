import csv
import hashlib
from pathlib import Path
from collections import defaultdict

CLEAN_ROOT = Path("data/1_cleaned")
IGNORE_DIR_NAMES = {"history"}  # skip archives

OUT_GROUPS = CLEAN_ROOT / "duplicate_groups.csv"
OUT_SUMMARY = CLEAN_ROOT / "duplicate_summary.csv"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA256 of a file without loading it all into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_md_files(root: Path):
    for p in root.rglob("*.md"):
        # skip anything inside ignored directories
        if any(parent.name.lower() in IGNORE_DIR_NAMES for parent in p.parents):
            continue
        yield p


def main():
    files = list(iter_md_files(CLEAN_ROOT))
    print(f"Markdown files scanned: {len(files)}")

    # Hash -> list[Path]
    groups = defaultdict(list)

    for i, p in enumerate(files, start=1):
        try:
            digest = sha256_file(p)
            groups[digest].append(p)
        except Exception as e:
            print(f" Failed to hash {p}: {e}")

        if i % 200 == 0:
            print(f"…hashed {i}/{len(files)}")

    dup_groups = {h: ps for h, ps in groups.items() if len(ps) > 1}
    total_dup_files = sum(len(ps) for ps in dup_groups.values())
    print(f"\nDuplicate groups found: {len(dup_groups)}")
    print(f"Files that are in duplicate groups: {total_dup_files}")

    # Assign stable group IDs (largest groups first)
    sorted_dups = sorted(dup_groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    # Write summary (one row per group)
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "sha256", "count"])
        for idx, (h, ps) in enumerate(sorted_dups, start=1):
            w.writerow([idx, h, len(ps)])

    # Write groups (one row per file)
    with OUT_GROUPS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "sha256", "file_path", "size_bytes"])
        for idx, (h, ps) in enumerate(sorted_dups, start=1):
            for p in ps:
                w.writerow([idx, h, str(p), p.stat().st_size])

    # Print biggest groups
    if sorted_dups:
        print("\nTop 10 duplicate groups:")
        for idx, (h, ps) in enumerate(sorted_dups[:10], start=1):
            print(f"  #{idx:02d}  count={len(ps)}  sha256={h[:12]}…")
    else:
        print("\nNo exact duplicates found.")

    print(f"\nWrote: {OUT_SUMMARY}")
    print(f"Wrote: {OUT_GROUPS}")


if __name__ == "__main__":
    main()