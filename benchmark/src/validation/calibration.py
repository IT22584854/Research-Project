"""
Calibration & Statistical Validation for Research-Grade Evaluation

This module provides tools to validate that your evaluation metrics
correlate with human expert judgments, enabling publication-ready results.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Container for calibration results"""
    spearman_rho: float
    spearman_pvalue: float
    pearson_r: float
    pearson_pvalue: float
    mean_absolute_error: float
    rmse: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    bootstrap_replicates: int
    sample_size: int


class MetricCalibrator:
    """
    Calibrate automatic metrics against human expert judgments.
    
    Workflow:
    1. Collect 100-200 responses with human expert ratings
    2. Compute automatic metrics for these responses
    3. Run calibration to compute correlation and error metrics
    4. Use calibration curve to adjust score scaling if needed
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Parameters
        ----------
        confidence_level : float
            CI level for bootstrap (e.g., 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        self.calibration_data = []
    
    def add_sample(self, auto_score: float, human_score: float, metadata: Optional[Dict] = None):
        """
        Add a calibration sample.
        
        Parameters
        ----------
        auto_score : float
            Score from automatic metric (0-1)
        human_score : float
            Score from human expert (0-1)
        metadata : dict, optional
            Additional info (question, domain, etc.)
        """
        if not (0 <= auto_score <= 1 and 0 <= human_score <= 1):
            logger.warning(f"Scores out of range: auto={auto_score}, human={human_score}")
            return
        
        self.calibration_data.append({
            "auto": auto_score,
            "human": human_score,
            "metadata": metadata or {}
        })
    
    def load_from_jsonl(self, path: str):
        """Load calibration samples from JSONL file"""
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.add_sample(
                    auto_score=item["auto_score"],
                    human_score=item["human_score"],
                    metadata=item.get("metadata")
                )
    
    def save_to_jsonl(self, path: str):
        """Save calibration samples to JSONL file"""
        with open(path, 'w') as f:
            for item in self.calibration_data:
                f.write(json.dumps(item) + '\n')
    
    def calibrate(self) -> CalibrationMetrics:
        """
        Compute calibration metrics.
        
        Returns
        -------
        CalibrationMetrics
            Correlation scores and error metrics
        """
        if len(self.calibration_data) < 10:
            raise ValueError(f"Need at least 10 samples, got {len(self.calibration_data)}")
        
        auto_scores = np.array([s["auto"] for s in self.calibration_data])
        human_scores = np.array([s["human"] for s in self.calibration_data])
        
        # Spearman rank correlation (robust to outliers)
        spearman_rho, spearman_pval = stats.spearmanr(auto_scores, human_scores)
        
        # Pearson correlation (assumes linear relationship)
        pearson_r, pearson_pval = stats.pearsonr(auto_scores, human_scores)
        
        # Error metrics
        errors = auto_scores - human_scores
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # Bootstrap confidence intervals
        ci_lower, ci_upper = self._bootstrap_ci(auto_scores, human_scores)
        
        return CalibrationMetrics(
            spearman_rho=float(spearman_rho),
            spearman_pvalue=float(spearman_pval),
            pearson_r=float(pearson_r),
            pearson_pvalue=float(pearson_pval),
            mean_absolute_error=float(mae),
            rmse=float(rmse),
            confidence_interval_lower=float(ci_lower),
            confidence_interval_upper=float(ci_upper),
            bootstrap_replicates=1000,
            sample_size=len(self.calibration_data)
        )
    
    def _bootstrap_ci(self, auto_scores: np.ndarray, human_scores: np.ndarray, 
                      n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for Spearman rho"""
        n = len(auto_scores)
        bootstrap_rhos = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            rho, _ = stats.spearmanr(auto_scores[idx], human_scores[idx])
            bootstrap_rhos.append(rho)
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_rhos, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_rhos, (1 - alpha/2) * 100)
        
        return ci_lower, ci_upper
    
    def apply_calibration_curve(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration curve to adjust raw scores.
        
        This fits a polynomial (degree 2) to the auto-vs-human relationship
        and applies it to new scores.
        
        Parameters
        ----------
        raw_scores : np.ndarray
            Automatic metric scores to calibrate
        
        Returns
        -------
        np.ndarray
            Calibrated scores
        """
        if len(self.calibration_data) < 10:
            logger.warning("Insufficient calibration data; returning raw scores")
            return raw_scores
        
        auto_scores = np.array([s["auto"] for s in self.calibration_data])
        human_scores = np.array([s["human"] for s in self.calibration_data])
        
        # Fit polynomial
        coeffs = np.polyfit(auto_scores, human_scores, deg=2)
        poly = np.poly1d(coeffs)
        
        # Apply to raw scores and clip to [0, 1]
        calibrated = np.clip(poly(raw_scores), 0, 1)
        return calibrated
    
    def plot_calibration_curve(self, output_path: str = "calibration_curve.png"):
        """
        Generate calibration curve plot (requires matplotlib).
        
        Parameters
        ----------
        output_path : str
            Where to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping plot")
            return
        
        auto_scores = np.array([s["auto"] for s in self.calibration_data])
        human_scores = np.array([s["human"] for s in self.calibration_data])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(auto_scores, human_scores, alpha=0.6, s=50)
        
        # Calibration curve (diagonal if perfect)
        x = np.linspace(0, 1, 100)
        coeffs = np.polyfit(auto_scores, human_scores, deg=2)
        poly = np.poly1d(coeffs)
        ax.plot(x, poly(x), 'r-', label='Calibration curve', linewidth=2)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
        
        ax.set_xlabel('Automatic Metric Score', fontsize=12)
        ax.set_ylabel('Human Expert Score', fontsize=12)
        ax.set_title('Metric Calibration Curve', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        logger.info(f"Calibration curve saved to {output_path}")
        plt.close()


def compute_inter_rater_reliability(rater1_scores: List[float], 
                                    rater2_scores: List[float]) -> Dict[str, float]:
    """
    Compute inter-rater reliability metrics.
    
    Parameters
    ----------
    rater1_scores : List[float]
        Scores from rater 1
    rater2_scores : List[float]
        Scores from rater 2
    
    Returns
    -------
    dict
        Reliability metrics (Cohen's kappa, correlation, etc.)
    """
    r1 = np.array(rater1_scores)
    r2 = np.array(rater2_scores)
    
    # Intraclass correlation (ICC)
    # For scores on continuous scale
    within_subject_var = np.mean((r1 - r2) ** 2) / 2
    grand_mean = np.mean([r1, r2])
    between_subject_var = np.var([r1, r2])
    
    icc = (between_subject_var - within_subject_var) / (between_subject_var + within_subject_var)
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(r1, r2)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(r1, r2)
    
    # Mean absolute difference
    mae = np.mean(np.abs(r1 - r2))
    
    return {
        "icc": float(icc),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "mean_absolute_diff": float(mae)
    }


def effect_size(group_a: List[float], group_b: List[float]) -> Dict[str, float]:
    """
    Calculate effect size (Cohen's d) between two groups.
    
    Parameters
    ----------
    group_a : List[float]
        Scores for group A
    group_b : List[float]
        Scores for group B
    
    Returns
    -------
    dict
        Effect size metrics
    """
    a = np.array(group_a)
    b = np.array(group_b)
    
    # Cohen's d
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0
    
    # Hedge's g (corrected for small sample sizes)
    correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
    hedges_g = cohens_d * correction
    
    return {
        "cohens_d": float(cohens_d),
        "hedges_g": float(hedges_g),
        "interpretation": _interpret_effect_size(abs(cohens_d))
    }


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d"""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def publication_readiness_report(calibration_metrics: CalibrationMetrics, 
                                 output_path: str = "calibration_report.json") -> Dict:
    """
    Generate a publication-ready calibration report.
    
    Parameters
    ----------
    calibration_metrics : CalibrationMetrics
        Results from calibrator.calibrate()
    output_path : str
        Where to save the JSON report
    
    Returns
    -------
    dict
        Report dict
    """
    report = {
        "publication_readiness": {
            "metric_calibration": "COMPLETE",
            "correlation": {
                "spearman_rho": calibration_metrics.spearman_rho,
                "spearman_pvalue": calibration_metrics.spearman_pvalue,
                "pearson_r": calibration_metrics.pearson_r,
                "pearson_pvalue": calibration_metrics.pearson_pvalue,
                "ci_95": [
                    calibration_metrics.confidence_interval_lower,
                    calibration_metrics.confidence_interval_upper
                ],
                "sample_size": calibration_metrics.sample_size,
                "bootstrap_replicates": calibration_metrics.bootstrap_replicates
            },
            "error_metrics": {
                "mean_absolute_error": calibration_metrics.mean_absolute_error,
                "rmse": calibration_metrics.rmse
            },
            "recommendations": []
        }
    }
    
    # Add recommendations based on results
    if calibration_metrics.spearman_pvalue > 0.05:
        report["publication_readiness"]["recommendations"].append(
            "Correlation not statistically significant (p > 0.05). Consider collecting more calibration data."
        )
    
    if calibration_metrics.spearman_rho < 0.7:
        report["publication_readiness"]["recommendations"].append(
            "Spearman correlation < 0.7. Consider refining metric prompts or weights."
        )
    
    if calibration_metrics.mean_absolute_error > 0.15:
        report["publication_readiness"]["recommendations"].append(
            "Mean absolute error > 0.15. Consider applying calibration curve adjustment."
        )
    
    if not report["publication_readiness"]["recommendations"]:
        report["publication_readiness"]["status"] = "PUBLICATION_READY"
    else:
        report["publication_readiness"]["status"] = "NEEDS_IMPROVEMENT"
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Publication readiness report saved to {output_path}")
    return report
