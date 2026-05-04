import json
import wandb

from frameworks.our_framework import evaluate_our_framework
from frameworks.baseline_framework import evaluate_baseline
from frameworks.ragas_framework import evaluate_ragas
from frameworks.deepeval_framework import evaluate_deepeval


with open("dataset/qa_dataset.json") as f:
    dataset = json.load(f)


def run_framework(name):

    print(f"\nRunning framework: {name}")

    wandb.init(
        project="rag_framework_comparison",
        name=name
    )

    for sample in dataset:

        q = sample["question"]
        c = sample["context"]
        a = sample["answer"]
        gt = sample["ground_truth"]

        if name == "our_framework":
            metrics = evaluate_our_framework(q, c, a, gt)

        elif name == "baseline":
            metrics = evaluate_baseline(q, c, a, gt)

        elif name == "ragas":
            metrics = evaluate_ragas(q, c, a, gt)

        elif name == "deepeval":
            metrics = evaluate_deepeval(q, c, a, gt)

        print("Metrics:", metrics)
        wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":

    run_framework("our_framework")
    run_framework("baseline")
    run_framework("ragas")
    run_framework("deepeval")