from FlagEmbedding import BGEM3FlagModel

# Load once globally
model = BGEM3FlagModel(
    'BAAI/bge-m3',
    use_fp16=True  
)

def embed_texts(texts, batch_size=16):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embeddings.tolist()