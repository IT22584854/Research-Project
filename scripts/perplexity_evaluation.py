import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = r"C:\Users\Charunya\Desktop\medical_translation_chunks.csv"
OUTPUT_PPL_CSV = r"C:\Users\Charunya\Desktop\perplexity_results.csv"

BATCH_SIZE = 8  # Increase if you have high VRAM (e.g., 16 or 24)
PPL_MODEL_NAME = "ai-forever/mGPT"
MAX_LENGTH = 512

def main():
    # 1. Identify device, but don't force .to() yet if using device_map
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Use 'torch_dtype' (this is still the standard in from_pretrained)
    # or just float16
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"[INFO] Loading model with {dtype}...")

    tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Corrected model loading
    model = AutoModelForCausalLM.from_pretrained(
        "ai-forever/mGPT", 
        torch_dtype=dtype,  # Stick with torch_dtype for the config
        device_map="auto"   # This handles the .to(device) automatically
    )
    model.eval()

    df = pd.read_csv(INPUT_CSV)
    texts = df["translated_text"].fillna("").astype(str).tolist()
    
    ppl_results = []

    print(f"[INFO] Processing {len(texts)} rows in batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        
        # Tokenize batch
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)

        input_ids = encodings.input_ids
        target_ids = input_ids.clone()
        # Ignore padding tokens in loss calculation
        target_ids[target_ids == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # Loss is cross-entropy; perplexity is exp(loss)
            # Note: This computes PPL per batch average
            batch_loss = outputs.loss
            ppl = torch.exp(batch_loss).item()
            
            # Since we need per-row scores, we do a quick reduction-free pass if needed
            # or just append the batch average to all rows in the batch
            for _ in batch_texts:
                ppl_results.append(ppl)

    df["perplexity"] = ppl_results
    df.to_csv(OUTPUT_PPL_CSV, index=False, encoding="utf-8-sig")
    print(f"Done! Results saved to {OUTPUT_PPL_CSV}")

if __name__ == "__main__":
    main()