import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BATCH_ID = "batch_69a49ef24b4c8190b50fd3870cfdf10e"
OUT_DIR = Path("data/6_curation/validation_gpt5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

b = client.batches.retrieve(BATCH_ID)
print("Batch status:", b.status)
print("output_file_id:", b.output_file_id)
print("error_file_id:", b.error_file_id)

if b.output_file_id:
    data = client.files.content(b.output_file_id).read()
    (OUT_DIR / "batch_output.jsonl").write_bytes(data)
    print("Wrote:", OUT_DIR / "batch_output.jsonl")

if b.error_file_id:
    data = client.files.content(b.error_file_id).read()
    (OUT_DIR / "batch_errors.jsonl").write_bytes(data)
    print("Wrote:", OUT_DIR / "batch_errors.jsonl")