#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from dotenv import load_dotenv
from openai import OpenAI

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python check_batch.py <BATCH_ID>")

    load_dotenv()
    client = OpenAI()

    batch_id = sys.argv[1]
    b = client.batches.retrieve(batch_id)

    print("BATCH_ID:", b.id)
    print("STATUS:", b.status)
    print("OUTPUT_FILE_ID:", b.output_file_id)
    print("ERROR_FILE_ID:", b.error_file_id)

if __name__ == "__main__":
    main()