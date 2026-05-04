"""Centralized configuration for the multi-agent system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Always load from agents/.env regardless of working directory
_AGENTS_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_AGENTS_ROOT / ".env")

# === Model Configuration ===
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# === Paths ===
AGENTS_ROOT = _AGENTS_ROOT
PROJECT_ROOT = AGENTS_ROOT.parent
DATA_DIR = AGENTS_ROOT / "data"

# === ChromaDB Configuration ===
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(AGENTS_ROOT / "chroma_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "medical_info")

# === Retrieval Configuration ===
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
ENSEMBLE_DENSE_WEIGHT = float(os.getenv("ENSEMBLE_DENSE_WEIGHT", "0.5"))

# === Limits ===
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "2000"))
MAX_CRITIQUE_ATTEMPTS = int(os.getenv("MAX_CRITIQUE_ATTEMPTS", "2"))
MAX_REWRITE_ATTEMPTS = int(os.getenv("MAX_REWRITE_ATTEMPTS", "3"))

# === Retry Configuration ===
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "1.0"))

# === Tavily Web Search Configuration ===
TAVILY_INCLUDE_DOMAINS = os.getenv(
    "TAVILY_INCLUDE_DOMAINS", 
    "epid.gov.lk,health.gov.lk"
).split(",")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# === Supabase Configuration ===
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "sl_med_corpus")
SUPABASE_PAGE_SIZE = int(os.getenv("SUPABASE_PAGE_SIZE", "1000"))
SUPABASE_NON_NULL_COLUMN = os.getenv("SUPABASE_NON_NULL_COLUMN", "").strip()

# === Medical Disclaimer ===
MEDICAL_DISCLAIMER = os.getenv(
    "MEDICAL_DISCLAIMER",
    "\n\n⚕️ MEDICAL DISCLAIMER: This information is for educational purposes only and should not be considered as medical advice. Always consult with a qualified healthcare professional for medical concerns, diagnosis, or treatment."
)

# === Custom LLM Configuration ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
CUSTOM_LLM_API_KEY = os.getenv("CUSTOM_LLM_API_KEY", "")
CUSTOM_LLM_BASE_URL = os.getenv("CUSTOM_LLM_BASE_URL", "")
CUSTOM_LLM_MODEL = os.getenv("CUSTOM_LLM_MODEL", "Qwen/Qwen3-1.7B")
CUSTOM_LLM_MAX_TOKENS = int(os.getenv("CUSTOM_LLM_MAX_TOKENS", "512"))
CUSTOM_LLM_TEMPERATURE = float(os.getenv("CUSTOM_LLM_TEMPERATURE", "0.7"))
