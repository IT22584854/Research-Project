from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

BATCH_ID = "batch_69a9f20e7e208190b96bf4ac6161d6b6"

OUT_PATH = Path(r"D:\SL_Medical_Corpus\data\4_instruction\regen_work\batch_output.jsonl")
ERR_PATH = Path(r"D:\SL_Medical_Corpus\data\4_instruction\regen_work\batch_error.jsonl")

batch = client.batches.retrieve(BATCH_ID)
print("Status:", batch.status)

if batch.status != "completed":
    raise SystemExit("Batch not completed yet.")

if batch.output_file_id:
    data = client.files.content(batch.output_file_id).read()
    OUT_PATH.write_bytes(data)
    print("Saved output ->", OUT_PATH)

if batch.error_file_id:
    data = client.files.content(batch.error_file_id).read()
    ERR_PATH.write_bytes(data)
    print("Saved errors ->", ERR_PATH)