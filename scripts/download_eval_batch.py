#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BATCH_ID = "batch_69ab46cab5748190a7aecfa3b0c0c699"

OUT_PATH = Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\repair_work\gold_eval_repair_batch_output_v2.jsonl")
ERR_PATH = Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\repair_work\gold_eval_repair_batch_error_v2.jsonl")

def main():
    load_dotenv()
    client = OpenAI()

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

if __name__ == "__main__":
    main()