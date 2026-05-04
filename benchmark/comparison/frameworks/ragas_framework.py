from ragas.metrics import answer_similarity, context_precision
from ragas import evaluate
from datasets import Dataset

from sentence_transformers import SentenceTransformer
from ragas.embeddings import BaseRagasEmbeddings

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


# ---------------------------
# Local Embedding Model
# ---------------------------

embed_model = SentenceTransformer("intfloat/multilingual-e5-large")


class LocalEmbeddings(BaseRagasEmbeddings):

    def embed_documents(self, texts):
        return embed_model.encode(texts).tolist()

    def embed_query(self, text):
        return embed_model.encode([text])[0].tolist()

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


embeddings = LocalEmbeddings()


# ---------------------------
# Local LLM (NO OPENAI)
# ---------------------------

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

llm = HuggingFacePipeline(pipeline=pipe)


# ---------------------------
# Evaluation Function
# ---------------------------

def evaluate_ragas(question, context, answer, ground_truth):

    data = {
        "question": [question],
        "contexts": [[context]],
        "answer": [answer],
        "ground_truth": [ground_truth]
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[answer_similarity, context_precision],
        llm=llm,
        embeddings=embeddings
    )

    scores = result.scores

    return {
        "answer_similarity": float(scores[0]["answer_similarity"]),
        "context_precision": float(scores[0]["context_precision"])
    }