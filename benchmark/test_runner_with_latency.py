import json
import numpy as np
from evaluation import evaluate_agent_response
from datetime import datetime


with open("data/benchmarks/gold_questions.json", "r", encoding="utf-8") as f:
    gold_questions = json.load(f)

with open("data/benchmarks/mock_agent_reponses_with_timestamp.json", "r", encoding="utf-8") as f:
    agent_responses = json.load(f)

# Organize agent responses by question ID
agent_response_dict = {}
for r in agent_responses:
    agent_response_dict.setdefault(r["question_id"], []).append(r)


latency_results = []

print("--- EVALUATION RESULTS ---\n")

for question in gold_questions:
    qid = question["id"]
    user_question = question["question"]

    # All responses for this question
    responses = agent_response_dict.get(qid, [{"response": "No response", "user_timestamp": None, "agent_timestamp": None}])

    for resp_data in responses:
        agent_response = resp_data["response"]

        # use stored timestamps if available, else ---> fallback to current time
        user_ts_str = resp_data.get("user_timestamp")
        agent_ts_str = resp_data.get("agent_timestamp")

        if user_ts_str:
            t_user_entered = datetime.fromisoformat(user_ts_str.replace("Z", "+00:00")).timestamp()
        else:
            t_user_entered = None

        if agent_ts_str:
            t_agent_start = datetime.fromisoformat(agent_ts_str.replace("Z", "+00:00")).timestamp()
        else:
            t_agent_start = None

        # --- Evaluation step ---
        score, grade, metrics = evaluate_agent_response(agent_response, question)

        # --- Latency calculations ---
        if t_user_entered and t_agent_start:
            agent_latency = 0  # Agent latency not measured separately in mock, set 0
            eval_latency = 0   # Evaluation latency not measured in mock, set 0
            total_latency = t_agent_start - t_user_entered
        else:
            agent_latency = 0
            eval_latency = 0
            total_latency = 0

        # Print results
        print(f"Question ID: {qid}")
        print(f"Response: {agent_response}")
        print(f"Metrics: {metrics}")
        print(f"Final Score: {score:.3f}")
        print(f"Grade: {grade}")
        print(f"Latencies (s): Agent={agent_latency:.3f}, Eval={eval_latency:.3f}, Total={total_latency:.3f}")
        print("-" * 60)

        # Save latency info
        latency_results.append({
            "question_id": qid,
            "response": agent_response,
            "agent_latency": agent_latency,
            "eval_latency": eval_latency,
            "total_latency": total_latency,
            "score": score,
            "grade": grade
        })

# --- Summary Statistics ---
for key in ["agent_latency", "eval_latency", "total_latency"]:
    values = [x[key] for x in latency_results]
    print(f"{key} -> mean: {np.mean(values):.3f}s, max: {np.max(values):.3f}s, min: {np.min(values):.3f}s, p95: {np.percentile(values, 95):.3f}s")

# save results to JSON 
with open("outputs/latency_results.json", "w", encoding="utf-8") as f:
    json.dump(latency_results, f, indent=4)
