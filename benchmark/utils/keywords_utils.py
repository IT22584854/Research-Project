from supabase import create_client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def process_keywords(keyword_field):
    if not keyword_field:
        return []

    if isinstance(keyword_field, str):
        return [k.strip().lower() for k in keyword_field.split(",")]

    if isinstance(keyword_field, list):
        return [k.strip().lower() for k in keyword_field]

    return []


def fetch_keywords():
    response = supabase.table("sl_med_corpus").select("doc_id, keywords").execute()

    keyword_map = {}

    for row in response.data:
        keyword_map[row["doc_id"]] = process_keywords(row.get("keywords"))

    return keyword_map