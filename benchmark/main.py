from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import utils.observability as obs
from engine import EvaluationEngine

load_dotenv()

app = FastAPI(title="RAG Evaluation API")

engine = EvaluationEngine(
    config_path="config.yaml",
    index_name=os.getenv("PINECONE_INDEX")
)


class AgentResponse(BaseModel):
    question: str
    answer: str
    start_timestamp: float
    end_timestamp: float
    mode: str = "with_ground_truth"


# @app.post("/evaluate")
# def evaluate(response: AgentResponse):

#     result = engine.evaluate(
#         agent_response=response.dict(),
#         mode=response.mode
#     )

#     return result

@app.post("/evaluate/ground-truth")
def evaluate_ground_truth(response: AgentResponse):
    result = engine.evaluate(
        agent_response=response.dict(),
        mode="with_ground_truth"
    )
    return result


@app.post("/evaluate/high-risk")
def evaluate_high_risk(response: AgentResponse):
    result = engine.evaluate(
        agent_response=response.dict(),
        mode="high_risk_medical"
    )
    return result