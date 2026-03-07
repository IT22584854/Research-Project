import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BATCH_FILE = Path("data/6_curation/validation_gpt5/repair_batch_input.jsonl")
OUT_DIR = Path("data/6_curation/validation_gpt5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

uploaded = client.files.create(file=BATCH_FILE.open("rb"), purpose="batch")
batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h")

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
    (OUT_DIR / "repair_batch_output.jsonl").write_bytes(output)
    print("Wrote:", OUT_DIR / "repair_batch_output.jsonl")

if status.error_file_id:
    errors = client.files.content(status.error_file_id).read()
    (OUT_DIR / "repair_batch_errors.jsonl").write_bytes(errors)
    print("Wrote:", OUT_DIR / "repair_batch_errors.jsonl")

print("Done.")