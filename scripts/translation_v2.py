import pandas as pd

input_file = r"C:\Users\Charunya\Desktop\processed_tokens.csv"
output_file = r"C:\Users\Charunya\Desktop\final_unique_medical_corpus.csv"

# 1. Load data
df = pd.read_csv(input_file, encoding='latin1')

# 2. Pre-clean the text for better matching
# Strip leading/trailing spaces which often cause "fake" uniqueness
df['clean_text'] = df['clean_text'].astype(str).str.strip()

# 3. Drop Duplicates
# We keep the 'first' instance of every unique medical text
df_unique = df.drop_duplicates(subset=['clean_text'], keep='first')

# 4. Filter out any empty/placeholder rows
df_unique = df_unique[df_unique['clean_text'].str.len() > 10]

# 5. Final Count & Save
final_count = len(df_unique)
df_unique.to_csv(output_file, index=False)

print(f"--- Deduping Complete ---")
print(f"Original Row Count: {len(df)}")
print(f"Cleaned Row Count: {final_count}")
print(f"File saved to: {output_file}")