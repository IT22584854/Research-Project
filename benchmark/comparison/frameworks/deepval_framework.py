from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = AnswerRelevancyMetric(threshold=0.5)

def evaluate_deepeval(question, context, answer, ground_truth):

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=ground_truth,
        context=[context]
    )

    metric.measure(test_case)

    return {
        "answer_relevancy": metric.score
    }