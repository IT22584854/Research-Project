# class WeightedScorer:
#     def __init__(self, config: dict):
#         self.config = config

#     def score(self, metric_scores: dict, mode: str) -> float:
#         weights = self.config["weights"][mode]

#         total_score = 0.0

#         for metric_name, weight in weights.items():
#             metric_value = metric_scores.get(metric_name, 0.0)
#             total_score += metric_value * weight

#         return round(total_score, 4)


class WeightedScorer:
    def __init__(self, config: dict):
        self.config = config

    def score(self, metric_scores: dict, mode: str) -> float:

        if mode not in self.config["weights"]:
            raise ValueError(f"Invalid mode: {mode}")

        weights = self.config["weights"][mode]

        total_score = 0.0
        total_weight = 0.0

        for metric_name, weight in weights.items():

            if metric_name not in metric_scores:
                print(f"[WARNING] Missing metric: {metric_name} in mode: {mode}")

            metric_value = metric_scores.get(metric_name, 0.0)

            total_score += metric_value * weight
            total_weight += weight

        # Optional normalization safety (in case weights don't sum to 1)
        if total_weight > 0:
            total_score = total_score / total_weight

        return round(total_score, 4)