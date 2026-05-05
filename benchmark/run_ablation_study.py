#!/usr/bin/env python3
"""
Ablation Study Runner for Health Domain LLM Evaluation Benchmark

This script demonstrates how to use the ablation framework to understand
which components of the evaluation system are most important and how
sensitive the system is to parameter changes.

Usage:
    python run_ablation_study.py --config config.yaml --output results.json
    python run_ablation_study.py --metric-ablation --metrics factual_accuracy safety
    python run_ablation_study.py --weight-sensitivity --metric semantic_similarity
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
import numpy as np

# Add benchmark directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ablation.ablation_study import AblationStudy


def create_sample_responses(n_samples: int = 10) -> List[Dict]:
    """Create sample agent responses for testing."""
    return [
        {
            'question': f'What are the symptoms of disease {i+1}?',
            'answer': f'The symptoms of disease {i+1} include fever, fatigue, and other medical indicators. Always consult a healthcare professional for proper diagnosis and treatment.',
            'start_timestamp': 1000.0 + i * 0.1,
            'end_timestamp': 1001.0 + i * 0.1
        }
        for i in range(n_samples)
    ]


def run_metric_ablation_study(ablation_study: AblationStudy, metrics: List[str],
                            test_responses: List[Dict], output_path: str = None) -> Dict:
    """Run metric ablation study."""
    print(f"Running metric ablation study for metrics: {metrics}")
    print(f"Using {len(test_responses)} test responses")

    results = ablation_study.run_metric_ablation(metrics, test_responses)

    # Print summary
    print("\n=== METRIC ABLATION RESULTS ===")
    print(".3f")
    print("\nImpact of removing each metric:")
    for metric, change in results['relative_changes'].items():
        print("6s")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {output_path}")

    return results


def run_weight_sensitivity_study(ablation_study: AblationStudy, metric: str,
                               weight_range: tuple = (0.0, 1.0), n_points: int = 11,
                               test_responses: List[Dict] = None, output_path: str = None) -> Dict:
    """Run weight sensitivity study."""
    if test_responses is None:
        test_responses = create_sample_responses(5)

    print(f"Running weight sensitivity study for metric: {metric}")
    print(f"Testing weight range: {weight_range[0]} to {weight_range[1]}")
    print(f"Using {len(test_responses)} test responses")

    results = ablation_study.run_weight_sensitivity(
        metric, weight_range, n_points, test_responses
    )

    # Print summary
    print("\n=== WEIGHT SENSITIVITY RESULTS ===")
    sensitivity = results['sensitivity_metrics']
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {output_path}")

    return results


def run_comprehensive_study(ablation_study: AblationStudy, test_responses: List[Dict],
                          output_path: str = "comprehensive_ablation_results.json") -> Dict:
    """Run comprehensive ablation study."""
    print("Running comprehensive ablation study...")
    print(f"Using {len(test_responses)} test responses")

    results = ablation_study.run_combined_ablation(test_responses, output_path)

    # Print key findings
    print("\n=== COMPREHENSIVE STUDY RESULTS ===")

    # Most impactful metrics
    ablation = results['metric_ablation']
    impacts = ablation.get('relative_changes', {})
    most_impactful = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    print("\nMost impactful metrics (when removed):")
    for metric, impact in most_impactful:
        print("6s")

    # Recommendations
    recommendations = results.get('recommendations', {})
    critical = recommendations.get('critical_metrics', [])
    if critical:
        print("\nCritical metrics requiring attention:")
        for item in critical:
            print(f"  - {item['metric']}: {item['recommendation']}")

    robust = recommendations.get('robust_metrics', [])
    if robust:
        print("\nRobust metrics:")
        for item in robust:
            print(f"  - {item['metric']}: {item['recommendation']}")

    print(f"\nComplete results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for LLM evaluation benchmark")
    
    # Auto-detect benchmark directory
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(benchmark_dir, 'config.yaml')
    
    parser.add_argument('--config', default=default_config, help='Path to config file')
    parser.add_argument('--index', help='Pinecone index name (uses env var if not provided)')

    # Study type arguments
    parser.add_argument('--metric-ablation', action='store_true', help='Run metric ablation study')
    parser.add_argument('--weight-sensitivity', action='store_true', help='Run weight sensitivity study')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive study')

    # Parameters
    parser.add_argument('--metrics', nargs='+', help='Metrics to ablate (for --metric-ablation)')
    parser.add_argument('--metric', help='Target metric for weight sensitivity')
    parser.add_argument('--weight-range', nargs=2, type=float, default=[0.0, 1.0],
                       help='Weight range for sensitivity study (min max)')
    parser.add_argument('--n-points', type=int, default=11, help='Number of weight points to test')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of test samples to generate')
    parser.add_argument('--output', help='Output path for results JSON')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.metric_ablation, args.weight_sensitivity, args.comprehensive]):
        print("Error: Must specify --metric-ablation, --weight-sensitivity, or --comprehensive")
        sys.exit(1)

    # Initialize ablation study
    try:
        ablation_study = AblationStudy(args.config, args.index)
        print("Ablation study initialized successfully")
    except Exception as e:
        print(f"Error initializing ablation study: {e}")
        sys.exit(1)

    # Generate test responses
    test_responses = create_sample_responses(args.n_samples)

    # Run requested study
    if args.metric_ablation:
        if not args.metrics:
            # Use all available metrics
            available_metrics = list(ablation_study.base_config['weights']['with_ground_truth'].keys())
            args.metrics = available_metrics

        run_metric_ablation_study(ablation_study, args.metrics, test_responses, args.output)

    elif args.weight_sensitivity:
        if not args.metric:
            print("Error: --metric required for weight sensitivity study")
            sys.exit(1)

        run_weight_sensitivity_study(
            ablation_study, args.metric, tuple(args.weight_range),
            args.n_points, test_responses, args.output
        )

    elif args.comprehensive:
        output_path = args.output or "comprehensive_ablation_results.json"
        run_comprehensive_study(ablation_study, test_responses, output_path)


if __name__ == "__main__":
    main()