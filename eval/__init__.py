"""MindGuard Benchmark - Evaluation Suite

This package provides tools for evaluating AI model safety calibration
in mental health contexts using the MindGuard DSpec-1200 benchmark.

Modules
-------
metrics
    Core metric calculations: calibration accuracy, consistency score,
    critical miss rate, over-escalation rate, context stability, and
    composite safety score.
automated_classifier
    Two-stage response classification system with feature extraction and
    rule-based classification into response levels R1-R5.
evaluate_model
    Main evaluation entry point with CLI support for running the full
    MindGuard evaluation pipeline.
"""

from eval.metrics import (
    calibration_accuracy,
    compute_all_metrics,
    consistency_score,
    context_stability,
    critical_miss_rate,
    over_escalation_rate,
    safety_score,
)
from eval.automated_classifier import (
    classify_batch,
    classify_response,
    extract_features,
    flag_problematic_patterns,
)
from eval.evaluate_model import (
    evaluate,
    generate_report,
    load_dataset,
    load_predictions,
)

__all__ = [
    # metrics
    "calibration_accuracy",
    "compute_all_metrics",
    "consistency_score",
    "context_stability",
    "critical_miss_rate",
    "over_escalation_rate",
    "safety_score",
    # automated_classifier
    "classify_batch",
    "classify_response",
    "extract_features",
    "flag_problematic_patterns",
    # evaluate_model
    "evaluate",
    "generate_report",
    "load_dataset",
    "load_predictions",
]
