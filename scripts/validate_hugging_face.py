from huggingface_hub import HfApi

api = HfApi()
info = api.repo_info("CharDiss/sri-lanka-medical-multiturn", repo_type="dataset")
print("private:", info.private)