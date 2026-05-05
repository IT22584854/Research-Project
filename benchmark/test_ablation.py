#!/usr/bin/env python3
"""
Test script for ablation studies framework.
Run this to verify the ablation implementation works correctly.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add benchmark directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the engine to avoid heavy imports during testing
class MockEvaluationEngine:
    def __init__(self, config_path, index_name=None):
        self.config_path = config_path
        self.index_name = index_name

    def evaluate(self, response, mode):
        # Return a mock score based on response content
        question_len = len(response.get('question', ''))
        answer_len = len(response.get('answer', ''))
        # Simple mock scoring logic
        base_score = 0.7
        if 'consult' in response.get('answer', '').lower():
            base_score += 0.1
        if 'doctor' in response.get('answer', '').lower():
            base_score += 0.1
        return {'score': min(1.0, base_score)}

# Monkey patch the lazy import to use mock
import src.ablation.ablation_study as ablation_module
ablation_module._import_evaluation_engine = lambda: MockEvaluationEngine

from src.ablation.ablation_study import AblationStudy


def test_ablation_framework():
    """Test the ablation framework with synthetic data."""
    print("Testing Ablation Studies Framework...")

    # Create temporary config for testing
    test_config = {
        'weights': {
            'with_ground_truth': {
                'factual_accuracy': 0.4,
                'semantic_similarity': 0.3,
                'safety': 0.2,
                'latency': 0.1
            }
        },
        'thresholds': {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'poor': 0.6,
            'critical': 0.5
        },
        'languages': ['en']
    }

    # Write temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(test_config, f)
        temp_config_path = f.name

    try:
        # Initialize ablation study
        ablation = AblationStudy(temp_config_path)
        print("✓ Ablation study initialized")

        # Create synthetic test responses
        test_responses = [
            {
                'question': 'What are vaccination benefits?',
                'answer': 'Vaccination prevents diseases and saves lives. Consult a doctor.',
                'start_timestamp': 1000.0,
                'end_timestamp': 1001.0
            },
            {
                'question': 'How to treat fever?',
                'answer': 'Rest and hydrate. Seek medical advice if persistent.',
                'start_timestamp': 1002.0,
                'end_timestamp': 1003.0
            }
        ]

        # Test metric removal
        print("Testing metric ablation...")
        ablation_results = ablation.run_metric_ablation(['safety'], test_responses)
        print("✓ Metric ablation completed")

        # Verify results structure
        assert 'base_scores' in ablation_results
        assert 'ablated_scores' in ablation_results
        assert 'score_changes' in ablation_results
        assert len(ablation_results['base_scores']) == len(test_responses)
        print("✓ Ablation results structure verified")

        # Test weight sensitivity
        print("Testing weight sensitivity...")
        sensitivity_results = ablation.run_weight_sensitivity(
            'factual_accuracy', (0.1, 0.7), 5, test_responses
        )
        print("✓ Weight sensitivity completed")

        # Verify sensitivity results
        assert 'weights_tested' in sensitivity_results
        assert 'mean_scores' in sensitivity_results
        assert 'sensitivity_metrics' in sensitivity_results
        assert len(sensitivity_results['weights_tested']) == 5
        print("✓ Sensitivity results structure verified")

        # Test config modification methods
        print("Testing config modification...")
        modified_config = ablation._remove_metric_from_config('safety')
        assert 'safety' not in modified_config['weights']['with_ground_truth']
        # Check weights sum to 1.0
        total_weight = sum(modified_config['weights']['with_ground_truth'].values())
        assert abs(total_weight - 1.0) < 0.001
        print("✓ Config modification verified")

        print("\n🎉 All ablation framework tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def test_ablation_script():
    """Test the command-line ablation script."""
    print("\nTesting ablation script...")

    # This would require setting up a full environment
    # For now, just check that the script exists and is importable
    try:
        import run_ablation_study
        print("✓ Ablation script is importable")
        return True
    except ImportError as e:
        print(f"❌ Script import failed: {e}")
        return False


if __name__ == "__main__":
    print("Running Ablation Framework Tests")
    print("=" * 40)

    success = True
    success &= test_ablation_framework()
    success &= test_ablation_script()

    if success:
        print("\n🎉 All tests passed! Ablation framework is ready to use.")
        print("\nTo run ablation studies:")
        print("  python run_ablation_study.py --comprehensive")
        print("  python run_ablation_study.py --metric-ablation --metrics factual_accuracy safety")
        print("  python run_ablation_study.py --weight-sensitivity --metric semantic_similarity")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)