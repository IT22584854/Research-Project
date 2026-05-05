# import os
# import json
# import hashlib
# from tqdm import tqdm
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv

# from supabase_loader import load_supabase_markdown
# from translation_loader import load_translations
# from embeddings import embed_texts
# from metadata_utils import detect_category

# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = "sl-health-index-with-translations"
# DIMENSION = 1024

# pc = Pinecone(api_key=PINECONE_API_KEY)

# if INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=DIMENSION,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )

# index = pc.Index(INDEX_NAME)


# # ------------------------
# # UTILS
# # ------------------------
# def hash_id(text):
#     return hashlib.md5(text.encode()).hexdigest()


# def parse_keywords(keywords_str):
#     if not keywords_str:
#         return []
#     try:
#         if isinstance(keywords_str, str):
#             return json.loads(keywords_str)
#         return keywords_str
#     except:
#         return []


# def clean_metadata(value):
#     if value is None:
#         return ""
#     if isinstance(value, list):
#         return [str(v) if v is not None else "" for v in value]
#     return value


# MAX_TEXT_LENGTH = 30000


# def truncate_text(text, max_length=MAX_TEXT_LENGTH):
#     if not text:
#         return ""
#     return text[:max_length] if len(text) > max_length else text


# def build_clean_embedding_text(clean_text, keywords):
#     """
#     Combine clean text + keywords for stronger semantic embedding
#     """
#     if not clean_text:
#         return ""

#     keyword_str = ", ".join(keywords) if keywords else ""
    
#     return f"""
# {clean_text}

# Keywords: {keyword_str}
# """


# # ------------------------
# # LOAD DATA
# # ------------------------
# docs = load_supabase_markdown()
# translations = load_translations()

# print("Docs:", len(docs))
# print("Translations:", len(translations))


# # ------------------------
# # TRANSLATION MAP
# # ------------------------
# translation_map = {}

# for t in translations:
#     doc_id = t.get("doc_id")
#     if doc_id:
#         translation_map[doc_id] = {
#             "si": t.get("translation_si"),
#             "ta": t.get("translation_ta"),
#             "en": t.get("translation_en")
#         }


# # ------------------------
# # BUILD VECTORS
# # ------------------------
# BATCH = 50
# all_vectors = []

# for doc in tqdm(docs):
#     doc_id = doc.get("doc_id") or doc.get("id")
#     if not doc_id:
#         continue

#     markdown = doc.get("markdown", "")
#     clean_text = doc.get("clean_text", "")

#     if not markdown and not clean_text:
#         continue

#     keywords = parse_keywords(doc.get("keywords", ""))

#     title = doc.get("title", "")
#     category = detect_category(doc_id)

#     # ------------------------
#     # 1️⃣ MARKDOWN EMBEDDING (EXISTING)
#     # ------------------------
#     if markdown:
#         emb = embed_texts([markdown])[0]

#         all_vectors.append({
#             "id": hash_id(doc_id + "_en_md"),
#             "values": emb,
#             "metadata": {
#                 "text": truncate_text(markdown),
#                 "source_id": doc_id,
#                 "lang": "en",
#                 "title": clean_metadata(title),
#                 "category": clean_metadata(category),
#                 "keywords": keywords,
#                 "content_type": "markdown",
#                 "is_translation": False
#             }
#         })

#     # ------------------------
#     # 2️⃣ CLEAN TEXT EMBEDDING (NEW)
#     # ------------------------
#     if clean_text:
#         enriched_clean = build_clean_embedding_text(clean_text, keywords)

#         emb = embed_texts([enriched_clean])[0]

#         all_vectors.append({
#             "id": hash_id(doc_id + "_en_clean"),
#             "values": emb,
#             "metadata": {
#                 "text": truncate_text(clean_text),
#                 "source_id": doc_id,
#                 "lang": "en",
#                 "title": clean_metadata(title),
#                 "category": clean_metadata(category),
#                 "keywords": keywords,
#                 "content_type": "clean_text",
#                 "is_translation": False
#             }
#         })

#     # ------------------------
#     # 3️⃣ TRANSLATIONS (UNCHANGED)
#     # ------------------------
#     trans = translation_map.get(doc_id, {})

#     for lang_code, trans_text in trans.items():
#         if trans_text:
#             emb = embed_texts([trans_text])[0]

#             all_vectors.append({
#                 "id": hash_id(doc_id + "_" + lang_code),
#                 "values": emb,
#                 "metadata": {
#                     "text": truncate_text(trans_text),
#                     "source_id": doc_id,
#                     "lang": lang_code,
#                     "title": clean_metadata(title),
#                     "category": clean_metadata(category),
#                     "keywords": keywords,
#                     "content_type": "translation",
#                     "is_translation": True
#                 }
#             })

#     # ------------------------
#     # UPSERT BATCH
#     # ------------------------
#     if len(all_vectors) >= BATCH:
#         index.upsert(vectors=all_vectors)
#         all_vectors = []


# # FINAL UPSERT
# if all_vectors:
#     index.upsert(vectors=all_vectors)

# print(f"✅ Done indexing {len(docs)} documents with clean_text + markdown + translations")



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

# ------------------------
# CONFIG
# ------------------------
BATCH = 50
MAX_TEXT_LENGTH = 1200   # 🔥 CRITICAL FIX (prevents 40KB Pinecone error)
MAX_KEYWORDS = 10


# ------------------------
# UTILS
# ------------------------
def hash_id(text):
    return hashlib.md5(text.encode()).hexdigest()


def parse_keywords(keywords_str):
    if not keywords_str:
        return []
    try:
        if isinstance(keywords_str, str):
            return json.loads(keywords_str)
        return keywords_str
    except:
        return []


def clean_metadata(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return [str(v)[:50] for v in value if v is not None][:MAX_KEYWORDS]
    return str(value)[:200]


def truncate_text(text, max_length=MAX_TEXT_LENGTH):
    if not text:
        return ""
    return text[:max_length]


def build_clean_embedding_text(clean_text, keywords):
    """
    Combine clean text + keywords for better semantic embedding
    (NOT stored in Pinecone metadata)
    """
    keyword_str = ", ".join(keywords[:MAX_KEYWORDS]) if keywords else ""
    return f"{clean_text}\n\nKeywords: {keyword_str}"


def safe_metadata(doc_id, text, title, category, keywords, lang, content_type, is_translation):
    """
    🔥 CRITICAL: Keep metadata SMALL (<40KB limit)
    """
    return {
        "source_id": str(doc_id),
        "text": truncate_text(text),   # HARD LIMIT
        "title": clean_metadata(title),
        "category": clean_metadata(category),
        "keywords": keywords[:MAX_KEYWORDS],
        "lang": lang,
        "content_type": content_type,
        "is_translation": is_translation
    }


# ------------------------
# LOAD DATA
# ------------------------
docs = load_supabase_markdown()
translations = load_translations()

print("Docs:", len(docs))
print("Translations:", len(translations))


# ------------------------
# TRANSLATION MAP
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
# BUILD VECTORS
# ------------------------
all_vectors = []

for doc in tqdm(docs):
    doc_id = doc.get("doc_id") or doc.get("id")
    if not doc_id:
        continue

    markdown = doc.get("markdown", "")
    clean_text = doc.get("clean_text", "")

    if not markdown and not clean_text:
        continue

    keywords = parse_keywords(doc.get("keywords", ""))
    title = doc.get("title", "")
    category = detect_category(doc_id)

    # ------------------------
    # 1️⃣ MARKDOWN EMBEDDING
    # ------------------------
    if markdown:
        emb = embed_texts([markdown])[0]

        all_vectors.append({
            "id": hash_id(doc_id + "_en_md"),
            "values": emb,
            "metadata": safe_metadata(
                doc_id,
                markdown,
                title,
                category,
                keywords,
                "en",
                "markdown",
                False
            )
        })

    # ------------------------
    # 2️⃣ CLEAN TEXT EMBEDDING
    # ------------------------
    if clean_text:
        enriched_clean = build_clean_embedding_text(clean_text, keywords)
        emb = embed_texts([enriched_clean])[0]

        all_vectors.append({
            "id": hash_id(doc_id + "_en_clean"),
            "values": emb,
            "metadata": safe_metadata(
                doc_id,
                clean_text,
                title,
                category,
                keywords,
                "en",
                "clean_text",
                False
            )
        })

    # ------------------------
    # 3️⃣ TRANSLATIONS
    # ------------------------
    trans = translation_map.get(doc_id, {})

    for lang_code, trans_text in trans.items():
        if trans_text:
            emb = embed_texts([trans_text])[0]

            all_vectors.append({
                "id": hash_id(doc_id + "_" + lang_code),
                "values": emb,
                "metadata": safe_metadata(
                    doc_id,
                    trans_text,
                    title,
                    category,
                    keywords,
                    lang_code,
                    "translation",
                    True
                )
            })

    # ------------------------
    # UPSERT BATCH
    # ------------------------
    if len(all_vectors) >= BATCH:
        index.upsert(vectors=all_vectors)
        all_vectors = []


# FINAL UPSERT
if all_vectors:
    index.upsert(vectors=all_vectors)

print(f"✅ Done indexing {len(docs)} documents (SAFE METADATA VERSION)")