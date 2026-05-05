import os
import yaml
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()

# Lazy imports to allow mocking in tests
def _import_evaluation_engine():
    # Import from benchmark directory
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.dirname(os.path.dirname(current_dir))
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)

    from engine import EvaluationEngine
    return EvaluationEngine

def _import_weighted_scorer():
    from src.scorer import WeightedScorer
    return WeightedScorer


class AblationStudy:
    """
    Framework for systematic ablation studies to understand evaluation sensitivity
    and component importance in the benchmark system.
    """

    def __init__(self, base_config_path: str, index_name: Optional[str] = None):
        """
        Initialize ablation study with base configuration.

        Args:
            base_config_path: Path to the base config.yaml file
            index_name: Pinecone index name (optional, will use env var if not provided)
        """
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)

        self.index_name = index_name or os.getenv("PINECONE_INDEX")
        if not self.index_name:
            raise ValueError(
                "Pinecone index name is missing. Set PINECONE_INDEX env var or pass --index to run_ablation_study.py."
            )

        self.base_scorer = _import_weighted_scorer()(self.base_config)

        # Initialize evaluation engine for running modified configs
        self.engine = _import_evaluation_engine()(base_config_path, self.index_name)

    def run_metric_ablation(self, metrics_to_remove: List[str], test_responses: List[Dict]) -> Dict[str, Any]:
        """
        Test impact of removing individual metrics from evaluation.

        Args:
            metrics_to_remove: List of metric names to remove
            test_responses: List of agent response dicts to evaluate

        Returns:
            Dict containing ablation results with score changes
        """
        results = {
            'base_scores': [],
            'ablated_scores': {},
            'score_changes': {},
            'relative_changes': {}
        }

        # Get baseline scores
        print("Running baseline evaluation...")
        for response in test_responses:
            result = self.engine.evaluate(response, mode='with_ground_truth')
            results['base_scores'].append(result.get('score', 0))

        base_mean = np.mean(results['base_scores'])

        # Test each metric removal
        for metric in metrics_to_remove:
            print(f"Testing ablation of metric: {metric}")

            # Create modified config without this metric
            modified_config = self._remove_metric_from_config(metric)

            # Create temporary config file for evaluation
            temp_config_path = self._create_temp_config(modified_config)

            try:
                # Create engine with modified config
                ablated_engine = _import_evaluation_engine()(temp_config_path, self.index_name)

                # Evaluate with modified config
                ablated_scores = []
                for response in test_responses:
                    result = ablated_engine.evaluate(response, mode='with_ground_truth')
                    ablated_scores.append(result.get('score', 0))

                ablated_mean = np.mean(ablated_scores)

                # Calculate impact
                score_change = ablated_mean - base_mean
                relative_change = (score_change / base_mean) * 100 if base_mean != 0 else 0

                results['ablated_scores'][metric] = ablated_scores
                results['score_changes'][metric] = score_change
                results['relative_changes'][metric] = relative_change

                print(".3f")

            finally:
                # Clean up temp config
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)

        return results

    def run_weight_sensitivity(self, target_metric: str, weight_range: Tuple[float, float] = (0.0, 1.0),
                             n_points: int = 21, test_responses: List[Dict] = None) -> Dict[str, Any]:
        """
        Test how sensitive final scores are to changes in a specific metric's weight.

        Args:
            target_metric: Name of metric to vary weight for
            weight_range: Tuple of (min_weight, max_weight)
            n_points: Number of weight values to test
            test_responses: List of agent responses to evaluate

        Returns:
            Dict containing sensitivity analysis results
        """
        if test_responses is None:
            # Use a small synthetic test set if none provided
            test_responses = self._create_synthetic_test_responses()

        results = {
            'weights_tested': [],
            'mean_scores': [],
            'std_scores': [],
            'sensitivity_curve': []
        }

        weights = np.linspace(weight_range[0], weight_range[1], n_points)

        print(f"Testing weight sensitivity for {target_metric}...")

        for weight in weights:
            print(".3f")

            # Create config with modified weight
            modified_config = self._modify_metric_weight(target_metric, weight)

            # Create temporary config file
            temp_config_path = self._create_temp_config(modified_config)

            try:
                # Evaluate with modified config
                modified_engine = _import_evaluation_engine()(temp_config_path, self.index_name)

                scores = []
                for response in test_responses:
                    result = modified_engine.evaluate(response, mode='with_ground_truth')
                    scores.append(result.get('score', 0))

                mean_score = np.mean(scores)
                std_score = np.std(scores)

                results['weights_tested'].append(weight)
                results['mean_scores'].append(mean_score)
                results['std_scores'].append(std_score)
                results['sensitivity_curve'].append({
                    'weight': weight,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'coefficient_of_variation': std_score / mean_score if mean_score != 0 else 0
                })

            finally:
                # Clean up temp config
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)

        # Calculate sensitivity metrics
        results['sensitivity_metrics'] = self._calculate_sensitivity_metrics(results)

        return results

    def run_combined_ablation(self, test_responses: List[Dict], output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive ablation study combining multiple analyses.

        Args:
            test_responses: List of agent responses to evaluate
            output_path: Optional path to save results as JSON

        Returns:
            Comprehensive ablation study results
        """
        results = {
            'timestamp': str(np.datetime64('now')),
            'base_config_summary': self._summarize_config(),
            'metric_ablation': {},
            'weight_sensitivity': {},
            'recommendations': {}
        }

        # Get all available metrics from config
        available_metrics = list(self.base_config['weights']['with_ground_truth'].keys())

        # Run metric ablation for all metrics
        print("Running comprehensive metric ablation...")
        ablation_results = self.run_metric_ablation(available_metrics, test_responses)
        results['metric_ablation'] = ablation_results

        # Run weight sensitivity for high-impact metrics
        high_impact_metrics = self._identify_high_impact_metrics(ablation_results)

        print("Running weight sensitivity analysis...")
        for metric in high_impact_metrics[:3]:  # Top 3 most impactful
            sensitivity_results = self.run_weight_sensitivity(metric, test_responses=test_responses)
            results['weight_sensitivity'][metric] = sensitivity_results

        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(ablation_results, results['weight_sensitivity'])

        # Save results if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {output_path}")

        return results

    def _remove_metric_from_config(self, metric_to_remove: str) -> Dict:
        """Create modified config with metric removed and weights renormalized."""
        modified_config = deepcopy(self.base_config)

        # Remove metric from weights
        weights = modified_config['weights']['with_ground_truth']
        if metric_to_remove in weights:
            del weights[metric_to_remove]

            # Renormalize remaining weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
                modified_config['weights']['with_ground_truth'] = weights

        return modified_config

    def _modify_metric_weight(self, target_metric: str, new_weight: float) -> Dict:
        """Create modified config with specific metric weight changed."""
        modified_config = deepcopy(self.base_config)
        weights = modified_config['weights']['with_ground_truth']

        if target_metric in weights:
            old_weight = weights[target_metric]
            weight_change = new_weight - old_weight

            # Adjust other weights to maintain total = 1.0
            other_metrics = [k for k in weights.keys() if k != target_metric]
            if other_metrics:
                weight_reduction = weight_change / len(other_metrics)
                for metric in other_metrics:
                    weights[metric] = max(0.0, weights[metric] - weight_reduction)

            weights[target_metric] = new_weight

            # Final normalization to ensure sum = 1.0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}

        modified_config['weights']['with_ground_truth'] = weights
        return modified_config

    def _create_temp_config(self, config: Dict) -> str:
        """Create temporary config file for evaluation."""
        import tempfile
        
        # Convert all numpy types to Python natives for safe YAML serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to Python natives."""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            else:
                return obj
        
        clean_config = convert_to_native(config)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(clean_config, temp_file)
        temp_file.close()
        return temp_file.name

    def _create_synthetic_test_responses(self, n_samples: int = 5) -> List[Dict]:
        """Create synthetic test responses for sensitivity analysis."""
        return [
            {
                'question': f'Sample medical question {i+1}',
                'answer': f'Sample medical answer {i+1} with some relevant information',
                'start_timestamp': 1000.0 + i,
                'end_timestamp': 1001.0 + i
            }
            for i in range(n_samples)
        ]

    def _calculate_sensitivity_metrics(self, sensitivity_results: Dict) -> Dict:
        """Calculate sensitivity metrics from weight variation results."""
        scores = np.array(sensitivity_results['mean_scores'])
        weights = np.array(sensitivity_results['weights_tested'])

        return {
            'score_range': float(np.max(scores) - np.min(scores)),
            'max_sensitivity_weight': float(weights[np.argmax(np.abs(np.gradient(scores)))]),
            'average_sensitivity': float(np.mean(np.abs(np.gradient(scores)))),
            'robustness_score': 1.0 - (np.std(scores) / np.mean(scores)) if np.mean(scores) != 0 else 0
        }

    def _identify_high_impact_metrics(self, ablation_results: Dict) -> List[str]:
        """Identify metrics with highest impact when removed."""
        impacts = ablation_results.get('relative_changes', {})
        sorted_metrics = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        return [metric for metric, _ in sorted_metrics]

    def _generate_recommendations(self, ablation_results: Dict, sensitivity_results: Dict) -> Dict:
        """Generate actionable recommendations from ablation results."""
        recommendations = {
            'critical_metrics': [],
            'robust_metrics': [],
            'weight_adjustments': [],
            'stability_concerns': []
        }

        # Identify critical metrics (high impact when removed)
        relative_changes = ablation_results.get('relative_changes', {})
        for metric, change in relative_changes.items():
            if abs(change) > 10:  # >10% change
                recommendations['critical_metrics'].append({
                    'metric': metric,
                    'impact': change,
                    'recommendation': 'High priority - consider increasing weight or improving implementation'
                })

        # Identify robust metrics (low sensitivity to weight changes)
        for metric, sensitivity in sensitivity_results.items():
            metrics_data = sensitivity.get('sensitivity_metrics', {})
            robustness = metrics_data.get('robustness_score', 0)
            if robustness > 0.8:  # Very robust
                recommendations['robust_metrics'].append({
                    'metric': metric,
                    'robustness_score': robustness,
                    'recommendation': 'Stable - weight changes have minimal impact'
                })

        return recommendations

    def _summarize_config(self) -> Dict:
        """Create summary of base configuration for reporting."""
        return {
            'evaluation_modes': list(self.base_config.get('weights', {}).keys()),
            'metrics': list(self.base_config.get('weights', {}).get('with_ground_truth', {}).keys()),
            'thresholds': self.base_config.get('thresholds', {}),
            'languages': self.base_config.get('languages', [])
        }