import wandb
import os
from dotenv import load_dotenv

load_dotenv()

wandb.login(key="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")

def init_run(framework_name):

    run = wandb.init(
        project="rag_framework_comparison",
        entity="it22280824-sliit",
        name=framework_name
    )

    return run


def log_metrics(metrics):

    wandb.log(metrics)