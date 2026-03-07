import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Point to the real location of the batch file
BATCH_FILE = Path(r"data/6_curation/idk_regen_curated885/curated885_idk_regen_batch_input.jsonl")

# Keep outputs in the same folder
OUT_DIR = Path(r"data/6_curation/idk_regen_curated885")
OUT_DIR.mkdir(parents=True, exist_ok=True)

uploaded = client.files.create(
    file=BATCH_FILE.open("rb"),
    purpose="batch"
)

batch = client.batches.create(
    input_file_id=uploaded.id,
    endpoint="/v1/responses",
    completion_window="24h"
)

print("Batch ID:", batch.id)

while True:
    status = client.batches.retrieve(batch.id)
    print("Status:", status.status)

    if status.status in ["completed", "failed", "expired", "cancelled"]:
        break

    time.sleep(10)

print("Final status:", status.status)
print("output_file_id:", status.output_file_id)
print("error_file_id:", status.error_file_id)

if status.output_file_id:
    output = client.files.content(status.output_file_id).read()
    out_path = OUT_DIR / "curated885_idk_regen_batch_output.jsonl"

    # guard: if a folder exists with this name, fall back
    if out_path.exists() and out_path.is_dir():
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = OUT_DIR / f"curated885_idk_regen_batch_output_{ts}.jsonl"

    out_path.write_bytes(output)
    print("Wrote:", out_path)

if status.error_file_id:
    errors = client.files.content(status.error_file_id).read()
    err_path = OUT_DIR / "curated885_idk_regen_batch_errors.jsonl"

    if err_path.exists() and err_path.is_dir():
        ts = time.strftime("%Y%m%d_%H%M%S")
        err_path = OUT_DIR / f"curated885_idk_regen_batch_errors_{ts}.jsonl"

    err_path.write_bytes(errors)
    print("Wrote:", err_path)

print("Done.")