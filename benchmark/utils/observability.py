import os
import wandb
import weave
from dotenv import load_dotenv

load_dotenv()

wandb.login(key=os.getenv("WANDB_API_KEY"))

weave.init("rag-evaluation-framework")