# 🧠 Agent Response Evaluation and Benchmark

This module is a **Retrieval-Augmented Generation (RAG) evaluation system** built with FastAPI.  
It is designed to evaluate LLM-generated answers using **multiple evaluation modes, hybrid metrics, and Supabase-stored interaction logs**.

It supports:
- Offline evaluation of LLM responses
- Batch evaluation of agent logs from Supabase
- Multi-metric scoring (factuality, safety, groundedness, web verification, etc.)
- Human + system evaluation workflows
- Frontend integration for reviewing and triggering evaluations

---

# 📌 Core Purpose

The system evaluates **agent-generated answers** stored in:


agent_turn_logs (Supabase table)


Each row contains:
- `user_message` → original question
- `assistant_message` → LLM response
- `rag_query` → optimized retrieval query (if available)
- `is_final_answer` → marks evaluatable responses

Only `is_final_answer = true` rows are used for evaluation.

---

# ⚙️ Evaluation Modes

The system supports **two major evaluation modes**:

---

## 🟢 1. `with_ground_truth`

### Purpose:
Used when evaluation is based on **retrieved context (Pinecone)**.

### How it works:
- Compares answer with:
  - retrieved chunks from Pinecone
- Evaluates correctness against known context

### Metrics:

| Metric | Description |
|------|--------|
| `semantic_similarity` | Similarity between question, answer, and retrieved chunks |
| `factual_accuracy` | Measures how many expected facts are present |
| `groundedness` | How well answer is supported by retrieved context |
| `safety` | Safety score (harmful content detection) |
| `latency` | Response generation time |

---

## 🔴 2. `high_risk_medical`

### Purpose:
Used for **critical domains (medical / safety-sensitive queries)**.

### Key Difference:
❌ Does NOT trust only retrieved documents  
✅ Uses **live web verification + reasoning checks**

### Metrics:

| Metric | Description |
|------|--------|
| `safe_web_verification` | Verifies atomic claims using web search |
| `self_consistency` | Checks consistency across multiple LLM samples |
| `uncertainty_expression` | Penalizes overconfident answers |
| `web_source_credibility` | Measures trustworthiness of web sources |
| `safety` | Toxicity / harmful content detection |
| `latency` | Execution time |
| `external_penalty` | Penalty for irrelevant or external noise |

### Safety Gates:
The system forces **critical rating (score = 0)** if:
- Safety score is too low
- Web verification confidence is low (< 40%)
- Overconfidence is too high

---

# 📊 Final Output (All Modes)

Each evaluation returns:


{
  "final_score": 0.87,
  "rating": "good",
  "metrics": {
    ...
  }
}

---
# 📊 Evaluation System

### 🏷️ Rating Scale

| Rating     | Score Range                  |
|------------|-----------------------------|
| excellent  | ≥ excellent threshold       |
| good       | ≥ good threshold            |
| acceptable | ≥ acceptable threshold      |
| poor       | ≥ poor threshold            |
| critical   | unsafe or failed evaluation |

---

## 🗄️ Supabase Integration

Each evaluation is automatically stored in the **`evaluations`** table.

### 📌 Stored Fields

#### Core Fields
- `mode`
- `question`
- `answer`
- `final_score`
- `rating`

#### 🧠 Ground Truth Metrics
- `semantic_similarity`
- `factual_accuracy`
- `groundedness`
- `claim_details`

#### ⚠️ High-Risk Metrics
- `safe_web_verification`
- `self_consistency`
- `uncertainty_expression`
- `web_source_credibility`
- `safe_details`
- `found_urls`

#### ⚙️ System Metrics
- `safety`
- `latency`
- `external_penalty`

---

## 🔁 API Routes

### 1. Get Evaluatable Turns
## 🔁 API Routes

1. Health / UI
GET /

Returns the frontend application.

GET /supabase/rows

Fetches raw rows from Supabase markdown loader.

Response:

{
  "rows": [...],
  "count": 120
}
2. Agent Turns
GET /agent-turns

Fetch all evaluatable agent turns.

Only returns:

is_final_answer = true

Query Params:

limit (default: 50)
offset (default: 0)

Response:

{
  "turns": [...],
  "count": 50,
  "limit": 50,
  "offset": 0
}
3. Single Turn Evaluation
🔴 High-Risk Medical Mode
POST /evaluate/turn/high-risk/{turn_id}

Evaluates a single turn using high-risk medical evaluation pipeline.

Query Params:

mode = high_risk_medical (default)
🟢 Ground Truth Mode
POST /evaluate/turn/ground-truth/{turn_id}

Evaluates a single turn using retrieval-based ground truth evaluation.

Query Params:

mode = with_ground_truth (default)
Response (both endpoints)
{
  "final_score": 0.87,
  "rating": "good",
  "metrics": {
    ...
  },
  "turn_id": 12,
  "session_id": "abc123",
  "turn_index": 3
}
4. Session-Level Evaluation
🔴 High-Risk Medical Mode
POST /evaluate/session/high-risk/{session_id}

Evaluates all final_answer turns in a session using medical safety pipeline.

🟢 Ground Truth Mode
POST /evaluate/session/ground-truth/{session_id}

Evaluates all final_answer turns using retrieved context grounding.

Response
{
  "session_id": "abc123",
  "mode": "high_risk_medical",
  "evaluated": 5,
  "results": [
    {
      "turn_id": 1,
      "turn_index": 0,
      "final_score": 0.82
    }
  ]
}
5. Batch Evaluation (Background Jobs)
🔴 High-Risk Batch
POST /evaluate/batch/high-risk

Queues background evaluation for pending turns.

🟢 Ground Truth Batch
POST /evaluate/batch/ground-truth

Queues background evaluation for pending turns.

Query Params:
limit (default: 20)
mode (optional override)
Response:
{
  "status": "queued",
  "mode": "high_risk_medical",
  "pending": 42,
  "queued": 20
}
6. Manual Evaluation (Testing / Debug)
🟢 Ground Truth Manual
POST /evaluate/ground-truth
🔴 High-Risk Manual
POST /evaluate/high-risk
Request Body
{
  "question": "string",
  "answer": "string",
  "start_timestamp": 0,
  "end_timestamp": 0,
  "mode": "with_ground_truth"
}
🧠 Notes
Only is_final_answer = true rows are evaluated from Supabase.
Batch jobs mark rows with evaluated_at after processing.
High-risk mode includes:
web verification
self-consistency checks
uncertainty penalties
Ground truth mode relies on:
Pinecone retrieval context
semantic similarity scoring
