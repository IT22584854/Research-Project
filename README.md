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

```json
{
  "final_score": 0.87,
  "rating": "good",
  "metrics": {
    ...
  }
}
## 📊 Evaluation System

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
```http
GET /agent-turns

Returns all records where:

is_final_answer = true

Used to populate the frontend evaluation dashboard.

2. Evaluate Single Turn
POST /evaluate/turn/{turn_id}

Evaluates a single Supabase record.

3. Evaluate Session
POST /evaluate/session/{session_id}

Evaluates all final answers in a session.

4. Batch Evaluation
POST /evaluate/batch

Runs background evaluation on multiple pending records.

5. Manual Evaluation
POST /evaluate/high-risk
POST /evaluate/ground-truth

Used for testing or external input.

🧑‍💻 Frontend Requirements
1. Evaluation Dashboard

Display a list containing:

question
answer
session_id
turn_index
latency
is_final_answer
2. Evaluate Button

Each row should include actions:

Evaluate (Ground Truth)
Evaluate (High Risk Medical)
3. Evaluation Result Viewer
📊 Metrics Panel
Final score
Rating badge (excellent, good, critical)
Full metric breakdown
🔍 Expandable Sections
claim_details
safe_details
retrieved_sources
found_urls
