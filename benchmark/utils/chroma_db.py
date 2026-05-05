import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from markdown_loader import load_markdown_corpus
from markdown_chunk import chunk_docs
from embeddings import embed_texts

# ----------------------------
# Connect to Local Chroma
# ----------------------------

client = chromadb.Client(
    Settings(
        persist_directory="./chroma_local_db"
    )
)

collection = client.get_or_create_collection(
    name="sl_health_collection"
)

# ----------------------------
# Load Markdown Corpus
# ----------------------------

CORPUS_DIR = "../data/ground_corpus"

docs = load_markdown_corpus(CORPUS_DIR)
print(f"Loaded {len(docs)} documents")

# ----------------------------
# Chunk Documents
# ----------------------------

chunks = chunk_docs(docs)
print(f"Created {len(chunks)} chunks")

# ----------------------------
# Embed + Store in Batches
# ----------------------------

UPLOAD_BATCH = 500
EMBED_BATCH = 16

for i in range(0, len(chunks), UPLOAD_BATCH):

    batch_chunks = chunks[i:i+UPLOAD_BATCH]

    texts = [c["text"] for c in batch_chunks]

    embeddings = embed_texts(texts, batch_size=EMBED_BATCH)

    collection.upsert(
        ids=[c["chunk_id"] for c in batch_chunks],
        documents=texts,
        metadatas=[{"source_id": c["source_id"]} for c in batch_chunks],
        embeddings=embeddings.tolist()
    )

    print(f"Uploaded {i + len(batch_chunks)} / {len(chunks)}")

# Persist database
client.persist()

print("✅ Successfully indexed to Local ChromaDB")