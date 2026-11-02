from flask import Blueprint, render_template, jsonify, request
import config
from services.model_registry import registry
from services.algorithm_comparison_data import (
    ALGORITHMS,
    DATA_EXPLORATION_TOOL,
    SIDE_BY_SIDE_PRESETS,
    DECISION_WIZARD,
    SCIKIT_LEARN_REFERENCES,
    get_all_algorithms,
    get_algorithm_by_id,
    get_preset_by_id
)

comparison_bp = Blueprint("comparison", __name__)

@comparison_bp.route("/comparison")
def model_comparison():
    metrics = registry.get_metrics()
    
    comparisons = {
        "tree_vs_rf": {
            "title": "Decision Tree vs Random Forest (Diabetes)",
            "description": "Compares an individual Gini Decision Tree against an ensemble of 100 bootstrapped trees on the identical stratified Pima Indians Diabetes test split.",
            "metrics": [
                {"metric": "Test Accuracy", "tree": f"{metrics.get('exp12', {}).get('comparison_decision_tree', {}).get('dt_accuracy', 0.72)*100:.2f}%", "rf": f"{metrics.get('exp12', {}).get('accuracy', 0.78)*100:.2f}%", "winner": "Random Forest"},
                {"metric": "F1-Score", "tree": f"{metrics.get('exp12', {}).get('comparison_decision_tree', {}).get('dt_f1', 0.65):.4f}", "rf": f"{metrics.get('exp12', {}).get('f1', 0.72):.4f}", "winner": "Random Forest"},
                {"metric": "Variance Sensitivity", "tree": "High (Sensitive to data noise)", "rf": "Low (Averaged across 100 trees)", "winner": "Random Forest"},
                {"metric": "Interpretability", "tree": "High (Direct IF-THEN rules)", "rf": "Low (Black-box ensemble)", "winner": "Decision Tree"}
            ],
            "dt_acc": round(metrics.get('exp12', {}).get('comparison_decision_tree', {}).get('dt_accuracy', 0.72)*100, 1),
            "rf_acc": round(metrics.get('exp12', {}).get('accuracy', 0.78)*100, 1)
        },
        "linear_vs_rbf_svm": {
            "title": "Linear SVM vs Non-Linear RBF SVM (Banknote)",
            "description": "Compares a flat hyperplane against an infinite-dimensional Gaussian radial basis function mapping on normalized banknote wavelet features.",
            "metrics": [
                {"metric": "Test Accuracy", "linear": f"{metrics.get('exp10', {}).get('accuracy', 0.98)*100:.2f}%", "rbf": f"{metrics.get('exp11', {}).get('accuracy', 1.0)*100:.2f}%", "winner": "RBF SVM"},
                {"metric": "Support Vectors Count", "linear": str(metrics.get('exp10', {}).get('support_vectors_count', 42)), "rbf": str(metrics.get('exp11', {}).get('support_vectors_count', 88)), "winner": "Linear (Sparseness)"},
                {"metric": "Boundary Shape", "linear": "Linear Hyperplane (w·x + b = 0)", "rbf": "Non-linear curved manifold", "winner": "Contextual"},
                {"metric": "Inference Latency", "linear": "Fast (Single dot product)", "rbf": "Moderate (Kernel distance per SV)", "winner": "Linear SVM"}
            ],
            "linear_acc": round(metrics.get('exp10', {}).get('accuracy', 0.98)*100, 1),
            "rbf_acc": round(metrics.get('exp11', {}).get('accuracy', 1.0)*100, 1)
        },
        "regression_suite": {
            "title": "Linear vs Multivariate vs Random Forest Regressors",
            "description": "Examines performance across regression algorithms and target problems.",
            "models": [
                {"name": "Simple Linear Regression", "target": "Placement (LPA)", "r2": f"{metrics.get('exp02', {}).get('placement', {}).get('r2', 0.82):.4f}", "mse": f"{metrics.get('exp02', {}).get('placement', {}).get('mse', 0.15):.4f}"},
                {"name": "Multiple Linear Regression", "target": "Placement (LPA)", "r2": f"{metrics.get('exp03', {}).get('cgpa_iq', {}).get('r2', 0.86):.4f}", "mse": f"{metrics.get('exp03', {}).get('cgpa_iq', {}).get('mse', 0.12):.4f}"},
                {"name": "Random Forest Regressor", "target": "Glucose (mg/dL)", "r2": f"{metrics.get('exp13', {}).get('r2', 0.45):.4f}", "mse": f"{metrics.get('exp13', {}).get('mse', 350.0):.4f}"}
            ]
        }
    }

    return render_template(
        "model_comparison.html",
        student_info=config.STUDENT_INFO,
        student=config.STUDENT_INFO,
        comparisons=comparisons,
        active_page="comparison"
    )

@comparison_bp.route("/algorithm-comparison")
def algorithm_comparison():
    """
    Dedicated theoretical Algorithm Comparison Hub.
    Compares all 12 ML predictive model implementations by working principles,
    features, advantages, limitations, best use cases, and input/output types.
    Includes Dataset Exploration as a separate non-predictive analytical tool.
    """
    return render_template(
        "algorithm_comparison.html",
        student_info=config.STUDENT_INFO,
        student=config.STUDENT_INFO,
        algorithms=ALGORITHMS,
        data_exploration_tool=DATA_EXPLORATION_TOOL,
        presets=SIDE_BY_SIDE_PRESETS,
        decision_wizard=DECISION_WIZARD,
        sklearn_references=SCIKIT_LEARN_REFERENCES,
        active_page="algo_hub"
    )

@comparison_bp.route("/api/algorithm-comparison")
def api_all_algorithms():
    """API endpoint providing structured algorithm comparison metadata."""
    return jsonify({
        "status": "success",
        "algorithms": ALGORITHMS,
        "data_exploration_tool": DATA_EXPLORATION_TOOL,
        "presets": SIDE_BY_SIDE_PRESETS,
        "decision_wizard": DECISION_WIZARD,
        "sklearn_references": SCIKIT_LEARN_REFERENCES
    })

@comparison_bp.route("/api/algorithm-comparison/<algo_id>")
def api_algorithm_detail(algo_id):
    """API endpoint providing details for a specific algorithm."""
    algo = get_algorithm_by_id(algo_id)
    if not algo:
        return jsonify({"status": "error", "message": f"Algorithm '{algo_id}' not found"}), 404
    return jsonify({"status": "success", "algorithm": algo})

