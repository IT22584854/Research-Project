import os
import hashlib
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from markdown_loader import load_markdown_corpus
from markdown_chunk import chunk_docs
from embeddings import embed_texts
from metadata_utils import detect_category

# ----------------------------
# LOAD ENVIRONMENT VARIABLES
# ----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "sl-health-index-with-text"
DIMENSION = 1024

UPLOAD_BATCH = 500
EMBED_BATCH = 16

# ----------------------------
# HELPER FUNCTION: HASH ID
# ----------------------------
def generate_safe_id(text: str) -> str:
    """
    Generate a safe ASCII ID for Pinecone by hashing.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# ----------------------------
# INITIALIZE PINECONE
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)
print("✅ Connected to Pinecone Cloud")

# ----------------------------
# LOAD AND CHUNK MARKDOWN CORPUS
# ----------------------------
CORPUS_DIR = "../data/ground_corpus"

docs = load_markdown_corpus(CORPUS_DIR)
print(f"Loaded {len(docs)} documents")

chunks = chunk_docs(docs)
for i, c in enumerate(chunks[:3]):
    print(f"--- Chunk {i} ---")
    print(c["text"][:500])

# ----------------------------
# EMBED AND UPLOAD TO PINECONE
# ----------------------------
for i in tqdm(range(0, len(chunks), UPLOAD_BATCH), desc="Uploading batches"):

    batch_chunks = chunks[i:i + UPLOAD_BATCH]
    texts = [c["text"] for c in batch_chunks]

    embeddings = embed_texts(texts, batch_size=EMBED_BATCH)

    vectors = []

    for chunk, embedding in zip(batch_chunks, embeddings):
        vectors.append({
            "id": generate_safe_id(chunk["chunk_id"]),
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "source_id": chunk["source_id"],
                "category": detect_category(chunk["source_id"])
            }
        })

    index.upsert(vectors=vectors)

print("✅ Successfully indexed to Pinecone Cloud")