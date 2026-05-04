import os
import logging
import yaml
import wandb
from dotenv import load_dotenv

from src.retrieval import PineconeRetriever
from src.scorer import WeightedScorer
from utils.supabase_db import store_evaluation

# ── with_ground_truth metrics ──────────────────────────────────────────────
from src.metrics.semantic_similarity import semantic_similarity
from src.metrics.factual_accuracy import factual_accuracy
from src.metrics.groundedness import groundedness_score

# ── high_risk_medical metrics (internet-verified, corpus-independent) ──────
from src.metrics.safe_web_verification import safe_web_verification
from src.metrics.self_consistency import self_consistency_score
from src.metrics.uncertainty_expression import uncertainty_expression_score
from src.metrics.source_credibility import source_credibility_score

# ── shared metrics ─────────────────────────────────────────────────────────
from src.metrics.safety import safety_score
from src.metrics.latency import latency_score
from src.metrics.external_detection import external_content_penalty

load_dotenv()

logger = logging.getLogger(__name__)
wandb.login(key=os.getenv("WANDB_API_KEY"))


class EvaluationEngine:

    def __init__(self, config_path, index_name):

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.retriever = PineconeRetriever(index_name)
        self.scorer    = WeightedScorer(self.config)

        # Single persistent W&B run — NOT reinit=True per call.
        # reinit=True floods the project with one-point runs and breaks
        # trend analysis. resume="allow" picks up the run across restarts.
        self._wandb_run = self._init_wandb()

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------

    def _init_wandb(self):
        try:
            return wandb.init(
                project="rag-evaluation",
                name="evaluation_run",
                resume="allow",
            )
        except Exception as e:
            logger.warning("W&B init failed: %s", e)
            return None

    def _log_wandb(self, payload: dict) -> None:
        if self._wandb_run is None:
            return
        try:
            wandb.log(payload)
        except Exception as e:
            logger.warning("W&B logging failed: %s", e)

    def finish(self) -> None:
        """Call on server shutdown to flush W&B."""
        if self._wandb_run is not None:
            self._wandb_run.finish()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def evaluate(self, agent_response: dict, mode: str):

        question = agent_response["question"]
        answer   = agent_response["answer"]
        start_ts = agent_response["start_timestamp"]
        end_ts   = agent_response["end_timestamp"]

        # 1️⃣ Retrieve from Pinecone
        # high_risk_medical: chunks feed the agent's answer and self_consistency
        # resampling only — NOT used to score factual correctness.
        retrieved_chunks, pinecone_scores, metadata = self.retriever.retrieve(
            question,
            top_k=5
        )

        metrics         = {}
        claim_details   = []
        factual_details = {}
        safe_details    = []
        found_urls      = []

        # ── COMMON ────────────────────────────────────────────────────────
        safety, safety_details = safety_score(str(answer), self.config)
        metrics["safety"]  = safety
        metrics["latency"] = latency_score(start_ts, end_ts)

        # ==================================================================
        # 🟢 MODE 1: WITH GROUND TRUTH  (unchanged)
        # ==================================================================
        if mode == "with_ground_truth":

            sim_scores = semantic_similarity(question, answer, retrieved_chunks)
            metrics["semantic_similarity"] = sim_scores["qa_similarity"]

            fa_score, fa_details = factual_accuracy(
                question=question,
                answer=answer,
                retrieved_chunks=retrieved_chunks
            )
            metrics["factual_accuracy"] = fa_score
            factual_details = fa_details

            groundedness, claim_details = groundedness_score(
                answer,
                retrieved_chunks
            )
            metrics["groundedness"] = groundedness

        # ==================================================================
        # 🔴 MODE 2: HIGH RISK MEDICAL  (hybrid — internet verification)
        #
        # Pinecone retrieval still drives the agent's answer but evaluation
        # verifies atomic claims against the live internet via SAFE
        # (Wei et al., NeurIPS 2024, arXiv:2403.18802), not against the
        # retrieved corpus.  This removes the core flaw: a wrong answer
        # consistent with wrong/stale Pinecone chunks can no longer score well.
        #
        # Metrics:
        #   safe_web_verification  — atomic facts verified by web search (0.35)
        #   self_consistency       — sampling divergence, corpus-independent (0.25)
        #   safety                 — hard pattern gate (0.20)
        #   uncertainty_expression — overconfidence penalty (0.10)
        #   web_source_credibility — authority of URLs found by SAFE (0.05)
        #   latency                — pipeline timing (0.05)
        # ==================================================================
        elif mode == "high_risk_medical":

            # 2️⃣ SAFE Web Verification
            # Decomposes answer into atomic facts, issues a real web search
            # per fact via the Anthropic web_search tool, and returns
            # SUPPORTED / NOT_SUPPORTED / CONTRADICTED per fact.
            # CONTRADICTED facts carry a 1.5x penalty over unsupported.
            safe_score, safe_details, found_urls = safe_web_verification(
                question,
                answer
            )
            metrics["safe_web_verification"] = safe_score

            # 3️⃣ Self Consistency
            # Generates N=3 stochastic re-answers and measures cross-sample
            # sentence similarity.  Hallucinated facts diverge; known facts
            # do not.  Catches confident hallucinations that happen to match
            # thin/low-quality web pages that SAFE might accept.
            sc_score, sc_details = self_consistency_score(
                question,
                answer,
                retrieved_chunks   # passed as LLM context for resampling only
            )
            metrics["self_consistency"] = sc_score

            # 4️⃣ Uncertainty Expression
            # Penalises overconfident phrasing ("you definitely have",
            # "guaranteed cure") on uncertain medical claims.
            uncertainty, uncertainty_details = uncertainty_expression_score(answer)
            metrics["uncertainty_expression"] = uncertainty

            # 5️⃣ Web Source Credibility
            # Scores the authority of the domains SAFE discovered during
            # verification.  A fact confirmed by who.int / health.gov.lk
            # is worth more than one confirmed by a blog.
            wsc_score, wsc_details = source_credibility_score(found_urls, self.config)
            metrics["web_source_credibility"] = wsc_score

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

        # ==================================================================
        # External Penalty  (all modes)
        # ==================================================================
        external_penalty, penalty_details = external_content_penalty(
            answer,
            retrieved_chunks
        )

        # ==================================================================
        # Final Score
        # ==================================================================
        final_score = self.scorer.score(metrics, mode)
        final_score = max(0.0, final_score - external_penalty)

        rating = self._rating_label(final_score)

        # ==================================================================
        # Safety Gates
        # ==================================================================
        if mode == "high_risk_medical":

            safety_min = self.config["thresholds"]["safety_minimum"]

            if (
                metrics["safety"]                    < safety_min
                or metrics["safe_web_verification"]  < 0.40   # < 40 % facts verified
                or metrics["uncertainty_expression"] < 0.30   # dangerously overconfident
            ):
                rating      = "critical"
                final_score = 0.0

        if mode == "with_ground_truth":

            if metrics["safety"] < self.config["thresholds"]["safety_minimum"]:
                rating      = "critical"
                final_score = 0.0

        # ==================================================================
        # W&B Logging  (single persistent run, not reinit per call)
        # ==================================================================
        self._log_wandb({
            **metrics,
            "external_penalty": external_penalty,
            "final_score":      final_score,
            "rating":           rating,
            "mode":             mode,
        })

        # ==================================================================
        # Supabase
        # ==================================================================
        db_row = {
            "mode":    mode,
            "question": str(question),
            "answer":   str(answer),

            # with_ground_truth columns
            "semantic_similarity": float(metrics.get("semantic_similarity", 0)),
            "factual_accuracy":    float(metrics.get("factual_accuracy",    0)),
            "factual_verdict":     factual_details.get("verdict"),
            "factual_reason":      factual_details.get("reason"),
            "groundedness":        float(metrics.get("groundedness", 0)),
            "claim_details": [
                {
                    "claim":      c.get("claim"),
                    "supported":  c.get("supported"),
                    "similarity": c.get("similarity"),
                }
                for c in claim_details
            ],

            # high_risk_medical columns
            "safe_web_verification":  metrics.get("safe_web_verification"),
            "self_consistency":       metrics.get("self_consistency"),
            "uncertainty_expression": metrics.get("uncertainty_expression"),
            "web_source_credibility": metrics.get("web_source_credibility"),
            "safe_details": [
                {
                    "fact":    d.get("fact"),
                    "verdict": d.get("verdict"),
                    "reason":  d.get("reason"),
                }
                for d in safe_details
            ],
            "found_urls": found_urls,

            # shared columns
            "retrieved_sources": [
                {
                    "category":  s.get("category"),
                    "source_id": s.get("source_id"),
                }
                for s in metadata
            ],
            "safety":           float(metrics.get("safety",  0)),
            "safety_details":   safety_details,
            "latency":          float(metrics.get("latency", 0)),
            "external_penalty": float(external_penalty),
            "final_score":      float(final_score),
            "rating":           str(rating),
        }

        store_evaluation(db_row)

        return {
            "status":           "success",
            "mode":             mode,
            "metrics":          metrics,
            # with_ground_truth detail keys
            "claim_details":    claim_details,
            "factual_details":  factual_details,
            # high_risk_medical detail keys
            "safe_details":     safe_details,
            "found_urls":       found_urls,
            # shared
            "safety_details":   safety_details,
            "external_penalty": external_penalty,
            "retrieved_sources": metadata,
            "final_score":      round(final_score, 4),
            "rating":           rating,
        }

    # ------------------------------------------------------------------

    def _rating_label(self, score):

        t = self.config["thresholds"]

        if score >= t["excellent"]:
            return "excellent"
        elif score >= t["good"]:
            return "good"
        elif score >= t["acceptable"]:
            return "acceptable"
        elif score >= t["poor"]:
            return "poor"
        else:
            return "critical"