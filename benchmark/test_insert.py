# from utils.supabase_db import store_evaluation

# sample_data = {
#     "question": "What is AI?",
#     "answer": "AI is Artificial Intelligence.",
#     "semantic_similarity": 0.9,
#     "factual_accuracy": 0.95,
#     "groundedness": 0.8,
#     "safety": 1.0,
#     "latency": 0.5,
#     "external_penalty": 0.0,
#     "final_score": 0.9,
#     "rating": "excellent"
# }

# store_evaluation(sample_data)

from utils.supabase_db import store_evaluation

store_evaluation({
    "question": "Test?",
    "answer": "Testing insert",
    "semantic_similarity": 0.5,
    "factual_accuracy": 0.5,
    "groundedness": 0.5,
    "safety": 1,
    "latency": 0.3,
    "external_penalty": 0,
    "final_score": 0.5,
    "rating": "acceptable"
})