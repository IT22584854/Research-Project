import os
import hashlib
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from supabase_loader import load_supabase_markdown
from translation_loader import load_translations
from markdown_chunk import chunk_docs
from embeddings import embed_texts
from metadata_utils import detect_category

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "sl-health-index-with-translations"
DIMENSION = 1024

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

def hash_id(text):
    return hashlib.md5(text.encode()).hexdigest()


# ------------------------
# LOAD DATA
# ------------------------
docs = load_supabase_markdown()
translations = load_translations()

print("Docs:", len(docs))
print("Translations:", len(translations))


# ------------------------
# CHUNK ENGLISH DOCS
# ------------------------
chunks = chunk_docs(docs)


# ------------------------
# BUILD TRANSLATION MAP
# ------------------------
translation_map = {}

for t in translations:
    key = (t["doc_id"], t["chunk_index"])
    translation_map[key] = {
        "si": t.get("translation_si"),
        "ta": t.get("translation_ta"),
        "en": t.get("translation_en")
    }


# ------------------------
# BUILD VECTORS
# ------------------------
BATCH = 200
EMBED_BATCH = 16

for i in tqdm(range(0, len(chunks), BATCH)):
    batch = chunks[i:i+BATCH]

    all_vectors = []

    for chunk in batch:
        doc_id = chunk["source_id"]
        chunk_idx = int(chunk["chunk_id"].split("_")[-1])

        translations = translation_map.get((doc_id, chunk_idx), {})

        texts = {
            "en": chunk["text"],
            "si": translations.get("si"),
            "ta": translations.get("ta")
        }

        for lang, text in texts.items():
            if not text:
                continue

            emb = embed_texts([text])[0]

            all_vectors.append({
                "id": hash_id(chunk["chunk_id"] + lang),
                "values": emb,
                "metadata": {
                    "text": text,
                    "chunk_id": chunk["chunk_id"],
                    "source_id": doc_id,
                    "lang": lang,
                    "category": detect_category(doc_id),
                    "is_translation": lang != "en"
                }
            })

    index.upsert(vectors=all_vectors)

print("✅ Done indexing multilingual vectors")