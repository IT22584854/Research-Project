from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
import numpy as np

bge_model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)

e5_model = SentenceTransformer(
    "intfloat/multilingual-e5-large"
)

def embed_bge(text):

    if isinstance(text, str):
        text = [text]

    return np.array(
        bge_model.encode(text)["dense_vecs"]
    )


def embed_e5(text):

    return np.array(
        e5_model.encode(text)
    )