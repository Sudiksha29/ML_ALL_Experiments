from flask import Blueprint, render_template
import config
from services.model_registry import registry
from services.history_service import get_recent_history

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/")
@dashboard_bp.route("/home", endpoint="home")
def index():
    experiments = config.EXPERIMENTS_META
    recent_history = get_recent_history(limit=6)

    # Actual verified models from registry
    required_model_keys = [
        "exp02_placement", "exp02_hours", "exp03_cgpa_iq", "exp03_realestate",
        "exp04_titanic", "exp04_placement", "exp05_student", "exp05_iris",
        "exp06_heart", "exp07_diabetes", "exp07_breast_cancer", "exp08_banknote",
        "exp09_mnb", "exp09_mnb_222", "exp10_linear_svm", "exp11_nonlinear_svm",
        "exp12_rf_diabetes", "exp13_rf_glucose"
    ]
    models_online_count = sum(1 for k in required_model_keys if registry.is_model_available(k))

    stats = {
        "total_experiments": len(experiments),
        "total_modules": len(experiments),
        "regression_count": sum(1 for e in experiments.values() if e["type"] == "regression"),
        "classification_count": sum(1 for e in experiments.values() if e["type"] in ["classification", "nlp"]),
        "models_trained": models_online_count,
        "datasets_available": 11,
        "all_models_online": models_online_count == len(required_model_keys)
    }

    return render_template(
        "dashboard.html",
        student=config.STUDENT_INFO,
        experiments=experiments,
        stats=stats,
        recent_history=recent_history,
        active_page="home"
    )

@dashboard_bp.route("/profile")
def profile():
    return render_template(
        "profile.html",
        student=config.STUDENT_INFO,
        active_page="profile"
    )

@dashboard_bp.route("/about")
def about():
    return render_template(
        "about.html",
        student=config.STUDENT_INFO,
        active_page="about"
    )
