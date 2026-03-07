import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

bucket = "instruction_ datasets"

files = [
    ("data/4_instruction/train_multiturn.jsonl", "multiturn/v1/train_multiturn.jsonl"),
    ("data/4_instruction/eval_multiturn.jsonl",  "multiturn/v1/eval_multiturn.jsonl"),
]

for local_path, remote_path in files:
    with open(local_path, "rb") as f:
        try:
            supabase.storage.from_(bucket).upload(
                path=remote_path,
                file=f,
                file_options={"content-type": "application/x-ndjson"},
            )
            print(f"✅ Uploaded: {remote_path}")
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "409" in msg:
                f.seek(0)
                supabase.storage.from_(bucket).update(
                    path=remote_path,
                    file=f,
                    file_options={"content-type": "application/x-ndjson"},
                )
                print(f"♻️ Updated: {remote_path}")
            else:
                raise

print("✅ Done uploading to Supabase Storage")