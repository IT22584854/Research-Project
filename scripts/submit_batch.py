#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def main():
    load_dotenv()
    client = OpenAI()

    input_path = Path(r"D:\SL_Medical_Corpus\data\4_instruction\regen_work\batch_input.jsonl")
    if not input_path.exists():
        raise SystemExit(f"batch_input.jsonl not found: {input_path}")

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