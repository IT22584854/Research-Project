from huggingface_hub import HfApi

repo_id = "CharDiss/sri-lanka-medical-multiturn"
api = HfApi()

# Create dataset repo (private=True if you want it private)
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

# Upload your JSONL files "as-is"
api.upload_file(
    path_or_fileobj="data/4_instruction/train_multiturn.jsonl",
    path_in_repo="train.jsonl",
    repo_id=repo_id,
    repo_type="dataset",
)

api.upload_file(
    path_or_fileobj="data/4_instruction/eval_multiturn.jsonl",
    path_in_repo="eval.jsonl",
    repo_id=repo_id,
    repo_type="dataset",
)

print("✅ Uploaded train + eval to:", repo_id)