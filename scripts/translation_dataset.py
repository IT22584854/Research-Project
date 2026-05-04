import pandas as pd
import numpy as np
import re
from collections import Counter
import math

# =========================================================
# CONFIGURATION
# =========================================================
INPUT_CSV = r"C:\Users\Charunya\Desktop\medical_translation_chunks_TEST100_with_chatgpt_nllb.csv"
OUTPUT_STATS_CSV = r"C:\Users\Charunya\Desktop\dataset_quality_audit.csv"

TEXT_COL = "translated_text"
LANG_COL = "target_lang"

# Settings for Lexical Diversity
MATTR_WINDOW = 100  # Window size for Moving Average TTR
MTLD_THRESHOLD = 0.72  # Standard threshold for MTLD

def tokenize(text):
    # Basic tokenization for multilingual text (works well for SI/TA/EN mixed)
    return re.findall(r'\w+', str(text).lower())

def calculate_mtld(tokens, threshold=0.72):
    def get_mtld_score(tks):
        if len(tks) < 1: return 0
        ttr_sum = 0
        count = 0
        current_types = set()
        
        for i, token in enumerate(tks):
            current_types.add(token)
            ttr = len(current_types) / (i + 1)
            if ttr < threshold:
                count += 1
                current_types = set()
        
        # Account for the remaining segment
        if count == 0: return len(tks) # or a default
        return len(tks) / count

    # MTLD is usually averaged from forward and backward passes
    forward = get_mtld_score(tokens)
    backward = get_mtld_score(tokens[::-1])
    return (forward + backward) / 2 if (forward + backward) > 0 else 0

def calculate_mattr(tokens, window_size=100):
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens) if tokens else 0
    
    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        ttrs.append(len(set(window)) / window_size)
    return np.mean(ttrs)

def calculate_entropy(tokens):
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((count/total) * math.log2(count/total) for count in counts.values())

def run_audit(df, lang_name):
    print(f"[INFO] Auditing {lang_name}...")
    # Combine all text for corpus-wide analysis
    all_text = " ".join(df[TEXT_COL].fillna("").astype(str))
    tokens = tokenize(all_text)
    
    if not tokens:
        return None

    # 1. Lexical Diversity
    unique_words = set(tokens)
    ttr = len(unique_words) / len(tokens)
    mattr = calculate_mattr(tokens, MATTR_WINDOW)
    mtld = calculate_mtld(tokens, MTLD_THRESHOLD)
    
    # 2. Diagnostic (Hapax Legomena)
    counts = Counter(tokens)
    hapax = [w for w, c in counts.items() if c == 1]
    
    # 3. Language Leakage (CMI Proxy)
    # Counts English tokens vs Total
    english_tokens = [w for w in tokens if re.match(r'^[a-z0-9]+$', w)]
    cmi = (len(english_tokens) / len(tokens)) * 100

    # 4. Complexity
    avg_sent_len = df[TEXT_COL].str.split().str.len().mean()
    entropy = calculate_entropy(tokens)

    return {
        "Language": lang_name,
        "Total_Tokens": len(tokens),
        "Unique_Words_NDW": len(unique_words),
        "TTR": round(ttr, 4),
        "MATTR": round(mattr, 4),
        "MTLD": round(mtld, 2),
        "Hapax_Legomena_Count": len(hapax),
        "English_Leakage_CMI_%": round(cmi, 2),
        "Wordform_Entropy": round(entropy, 4),
        "Avg_Sentence_Length": round(avg_sent_len, 2)
    }

def main():
    df = pd.read_csv(INPUT_CSV)
    
    results = []
    for lang in df[LANG_COL].unique():
        lang_df = df[df[LANG_COL] == lang]
        stats = run_audit(lang_df, lang)
        if stats:
            results.append(stats)
            
    # Save results
    audit_df = pd.DataFrame(results)
    audit_df.to_csv(OUTPUT_STATS_CSV, index=False)
    print(f"\n[DONE] Audit report saved to: {OUTPUT_STATS_CSV}")
    print(audit_df.to_string(index=False))

if __name__ == "__main__":
    main()