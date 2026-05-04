import csv
import shutil
from pathlib import Path
from collections import defaultdict

CLEAN_ROOT = Path("data/1_cleaned")
DUP_CSV = CLEAN_ROOT / "duplicate_groups.csv"
DUP_DEST = CLEAN_ROOT / "_duplicates"
MAP_OUT = CLEAN_ROOT / "deduped_map.csv"

def main():
    if not DUP_CSV.exists():
        raise FileNotFoundError(f"Missing {DUP_CSV}. Run find_duplicates.py first.")

    DUP_DEST.mkdir(parents=True, exist_ok=True)

    # group_id -> list of file paths
    groups = defaultdict(list)

    with DUP_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = int(row["group_id"])
            p = Path(row["file_path"])
            groups[gid].append(p)

    moved = 0
    with MAP_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "kept_path", "moved_path", "moved_to"])

        for gid in sorted(groups.keys()):
            files = sorted(groups[gid], key=lambda x: str(x).lower())
            if len(files) < 2:
                continue

            kept = files[0]

            for dup in files[1:]:
                if not dup.exists():
                    # already moved manually, or path mismatch
                    continue

                # preserve relative path under CLEAN_ROOT
                rel = dup.relative_to(CLEAN_ROOT)
                dest = DUP_DEST / rel
                dest.parent.mkdir(parents=True, exist_ok=True)

                # If destination already exists, add suffix
                if dest.exists():
                    dest = dest.with_name(dest.stem + "__dup" + dest.suffix)

                shutil.move(str(dup), str(dest))
                moved += 1
                w.writerow([gid, str(kept), str(dup), str(dest)])

    print(f"Done. Moved {moved} duplicate files into: {DUP_DEST}")
    print(f"Map written to: {MAP_OUT}")

if __name__ == "__main__":
    main()