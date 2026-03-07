#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def main():
    load_dotenv()
    client = OpenAI()

    input_path = Path(r"D:\SL_Medical_Corpus\data\7_gold_standard\repair_work\gold_eval_repair_batch_input_v2.jsonl")
    if not input_path.exists():
        raise SystemExit(f"batch input not found: {input_path}")

    uploaded = client.files.create(
        file=input_path.open("rb"),
        purpose="batch"
    )

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h"
    )

    print("BATCH_ID:", batch.id)
    print("STATUS:", batch.status)
    print("INPUT_FILE_ID:", uploaded.id)

if __name__ == "__main__":
    main()