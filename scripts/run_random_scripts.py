import os
import argparse
import pandas as pd
from tqdm import tqdm
from supabase import create_client, Client


def clean_value(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value if value else None


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--doc-id-col", default="doc_id")
    parser.add_argument("--text-col", default="clean_text")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY first.")

    supabase: Client = create_client(supabase_url, supabase_key)

    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    if args.doc_id_col not in df.columns:
        raise ValueError(f"Missing CSV column: {args.doc_id_col}")

    if args.text_col not in df.columns:
        raise ValueError(f"Missing CSV column: {args.text_col}")

    df = df[[args.doc_id_col, args.text_col]].copy()
    df[args.doc_id_col] = df[args.doc_id_col].astype(str).str.strip()
    df[args.text_col] = df[args.text_col].apply(clean_value)

    df = df[df[args.doc_id_col] != ""]
    df = df.drop_duplicates(subset=[args.doc_id_col], keep="last")

    df = df.dropna(subset=[args.doc_id_col])
    df[args.doc_id_col] = df[args.doc_id_col].astype(str).str.strip()
    df = df[df[args.doc_id_col] != ""]
    df = df[df[args.doc_id_col].str.lower() != "nan"]

    csv_doc_ids = set(df[args.doc_id_col].tolist())

    print(f"CSV rows after cleaning: {len(df)}")

    print("Fetching Supabase doc_ids...")

    supabase_rows = []
    page_size = 1000
    offset = 0

    while True:
        res = (
            supabase.table(args.table)
            .select(args.doc_id_col)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        rows = res.data or []
        supabase_rows.extend(rows)

        if len(rows) < page_size:
            break

        offset += page_size

    supabase_doc_ids = set(
        str(row[args.doc_id_col]).strip()
        for row in supabase_rows
        if row.get(args.doc_id_col) is not None
        and str(row[args.doc_id_col]).strip() != ""
        and str(row[args.doc_id_col]).strip().lower() != "nan"
    )

    matching_ids = csv_doc_ids & supabase_doc_ids
    csv_not_in_supabase = csv_doc_ids - supabase_doc_ids
    supabase_not_in_csv = supabase_doc_ids - csv_doc_ids

    print("\nSummary:")
    print(f"Matching doc_ids to update: {len(matching_ids)}")
    print(f"Supabase doc_ids missing in CSV, set clean_text=NULL: {len(supabase_not_in_csv)}")
    print(f"CSV doc_ids missing in Supabase, ignored/reported: {len(csv_not_in_supabase)}")

    if csv_not_in_supabase:
        with open("csv_doc_ids_not_in_supabase.txt", "w", encoding="utf-8") as f:
            for doc_id in sorted(map(str, csv_not_in_supabase)):
                f.write(doc_id + "\n")
        print("Report saved: csv_doc_ids_not_in_supabase.txt")

    if args.dry_run:
        print("\nDry run only. No Supabase rows changed.")
        return

    update_df = df[df[args.doc_id_col].isin(matching_ids)]

    print("\nUpdating matching Supabase rows...")

    updated = 0
    failed = 0

    for _, row in tqdm(update_df.iterrows(), total=len(update_df)):
        doc_id = row[args.doc_id_col]
        clean_text = row[args.text_col]

        try:
            supabase.table(args.table).update({
                args.text_col: clean_text
            }).eq(args.doc_id_col, doc_id).execute()

            updated += 1

        except Exception as e:
            failed += 1
            print(f"Failed update doc_id={doc_id}: {e}")

    print("\nSetting clean_text=NULL for Supabase rows not in CSV...")

    nulled = 0
    null_failed = 0

    for doc_id_batch in tqdm(list(chunk_list(list(supabase_not_in_csv), args.batch_size))):
        try:
            (
                supabase.table(args.table)
                .update({args.text_col: None})
                .in_(args.doc_id_col, doc_id_batch)
                .execute()
            )
            nulled += len(doc_id_batch)

        except Exception as e:
            null_failed += len(doc_id_batch)
            print(f"Failed NULL batch: {e}")

    print("\nDone.")
    print(f"Updated clean_text rows: {updated}")
    print(f"Failed updates: {failed}")
    print(f"Set clean_text=NULL rows: {nulled}")
    print(f"Failed NULL rows: {null_failed}")
    print(f"Ignored CSV doc_ids not in Supabase: {len(csv_not_in_supabase)}")


if __name__ == "__main__":
    main()