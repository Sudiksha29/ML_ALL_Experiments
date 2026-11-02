from flask import Blueprint, render_template, request, jsonify
import config
from services.history_service import get_recent_history, clear_history
from services.experiment_content import EXPERIMENT_DEEP_DIVE
from services.viva_guide_data import (
    COMMON_ML_CONCEPTS,
    COMPARISON_QUESTIONS,
    CODE_VIVA_QUESTIONS,
    MUST_KNOW_CODE_QUESTIONS,
    LAST_MINUTE_FORMULAS,
    TEST_YOUR_PREP_QUIZ
)

learning_bp = Blueprint("learning", __name__)

@learning_bp.route("/learning-center")
def learning_center():
    # Construct 13 algorithm viva topics directly from curriculum
    algo_topics = []
    for exp_id, meta in config.EXPERIMENTS_META.items():
        deep_dive = EXPERIMENT_DEEP_DIVE.get(exp_id, {})
        algo_topics.append({
            "exp_id": exp_id,
            "num": meta["num"],
            "title": f"EXP {meta['num']}: {meta['title']}",
            "algorithm": meta["algorithm"],
            "category": meta["category_badge"],
            "formula": deep_dive.get("formula", {}).get("equation", ""),
            "questions": deep_dive.get("viva_qa", [])
        })

    return render_template(
        "learning_center.html",
        student=config.STUDENT_INFO,
        common_concepts=COMMON_ML_CONCEPTS,
        algo_topics=algo_topics,
        comparison_questions=COMPARISON_QUESTIONS,
        code_questions=CODE_VIVA_QUESTIONS,
        must_know_code=MUST_KNOW_CODE_QUESTIONS,
        formulas=LAST_MINUTE_FORMULAS,
        quiz=TEST_YOUR_PREP_QUIZ,
        active_page="learning"
    )

@learning_bp.route("/model-playground")
def model_playground():
    return render_template(
        "model_playground.html",
        student=config.STUDENT_INFO,
        experiments=config.EXPERIMENTS_META,
        active_page="playground"
    )

@learning_bp.route("/history")
def history():
    history_items = get_recent_history(limit=100)
    return render_template(
        "history.html",
        student=config.STUDENT_INFO,
        history=history_items,
        active_page="history"
    )

@learning_bp.route("/api/history/clear", methods=["POST"])
def clear_history_api():
    success = clear_history()
    return jsonify({"status": "success" if success else "error"})
