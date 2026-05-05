import os
import json
import hashlib
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from supabase_loader import load_supabase_markdown
from translation_loader import load_translations
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


def parse_keywords(keywords_str):
    """Parse keywords from JSON string or return empty list"""
    if not keywords_str:
        return []
    try:
        if isinstance(keywords_str, str):
            return json.loads(keywords_str)
        return keywords_str
    except:
        return []


def clean_metadata(value):
    """Clean metadata value - convert None/null to empty string"""
    if value is None:
        return ""
    if isinstance(value, list):
        # Pinecone allows list of strings only
        return [str(v) if v is not None else "" for v in value]
    return value


MAX_TEXT_LENGTH = 30000  # Leave room for other metadata (40KB limit)


def truncate_text(text, max_length=MAX_TEXT_LENGTH):
    """Truncate text to fit within metadata size limit"""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length]


# ------------------------
# LOAD DATA
# ------------------------
docs = load_supabase_markdown()
translations = load_translations()

print("Docs:", len(docs))
print("Translations:", len(translations))


# ------------------------
# BUILD TRANSLATION MAP (by doc_id)
# ------------------------
translation_map = {}

for t in translations:
    doc_id = t.get("doc_id")
    if doc_id:
        translation_map[doc_id] = {
            "si": t.get("translation_si"),
            "ta": t.get("translation_ta"),
            "en": t.get("translation_en")
        }


# ------------------------
# BUILD VECTORS (full markdown, not chunked)
# ------------------------
BATCH = 50  # Smaller batch since full markdown is larger

all_vectors = []

for doc in tqdm(docs):
    doc_id = doc.get("doc_id") or doc.get("id")
    if not doc_id:
        continue
    
    # Get the full markdown
    markdown = doc.get("markdown", "")
    if not markdown:
        continue
    
    # Parse keywords from the document
    keywords = parse_keywords(doc.get("keywords", ""))
    
    # Get metadata
    title = doc.get("title", "")
    source_url = doc.get("source_url", "")
    source_type = doc.get("source_type", "")
    site = doc.get("site", "")
    category = detect_category(doc_id)
    
    # Original language
    lang = doc.get("language", "en")
    
    # Embed the full markdown (English)
    if markdown:
        emb = embed_texts([markdown])[0]
        
        all_vectors.append({
            "id": hash_id(doc_id + "_en"),
            "values": emb,
            "metadata": {
                "text": truncate_text(markdown),
                "source_id": doc_id,
                "lang": "en",
                "title": clean_metadata(title),
                "category": clean_metadata(category),
                "keywords": keywords if keywords else [],
                "is_translation": False
            }
        })
    
    # Add translations if available
    trans = translation_map.get(doc_id, {})
    for lang_code, trans_text in trans.items():
        if trans_text:
            emb = embed_texts([trans_text])[0]
            
            all_vectors.append({
                "id": hash_id(doc_id + "_" + lang_code),
                "values": emb,
                "metadata": {
                    "text": truncate_text(trans_text),
                    "source_id": doc_id,
                    "lang": lang_code,
                    "title": clean_metadata(title),
                    "category": clean_metadata(category),
                    "keywords": keywords if keywords else [],
                    "is_translation": True
                }
            })
    
    # Upsert in batches
    if len(all_vectors) >= BATCH:
        index.upsert(vectors=all_vectors)
        all_vectors = []

# Final batch
if all_vectors:
    index.upsert(vectors=all_vectors)

print(f"✅ Done indexing {len(docs)} documents with translations")