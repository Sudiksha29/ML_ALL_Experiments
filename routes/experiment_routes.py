from flask import Blueprint, render_template, request, abort
import config
from services.model_registry import registry
from services.experiment_content import get_experiment_deep_dive
from services.dataset_service import get_available_datasets, load_dataset_details

experiment_bp = Blueprint("experiments", __name__)

@experiment_bp.route("/experiments")
def list_experiments():
    # Directly open the interactive experiment execution workspace
    target = request.args.get("exp_id", "exp02")
    return experiment_detail(target)

@experiment_bp.route("/experiment/<exp_id>")
def experiment_detail(exp_id):
    meta = config.EXPERIMENTS_META.get(exp_id)
    if not meta:
        abort(404)

    metrics = registry.get_metrics(exp_id)
    all_experiments = config.EXPERIMENTS_META
    deep_dive = get_experiment_deep_dive(exp_id)

    # Find prev and next for smooth navigation
    keys = list(all_experiments.keys())
    idx = keys.index(exp_id)
    prev_exp = all_experiments[keys[idx - 1]] if idx > 0 else None
    next_exp = all_experiments[keys[idx + 1]] if idx < len(keys) - 1 else None

    # Dedicated interactive studio for Experiment 01 (EDA & Visualization)
    if exp_id == "exp01":
        selected_ds = request.args.get("dataset", "placement")
        datasets = get_available_datasets()
        ds_details = load_dataset_details(selected_ds)
        return render_template(
            "experiment_01_eda.html",
            student=config.STUDENT_INFO,
            exp=meta,
            datasets=datasets,
            selected_dataset=ds_details,
            deep_dive=deep_dive,
            prev_exp=prev_exp,
            next_exp=next_exp,
            all_experiments=all_experiments,
            active_page="experiments"
        )

    return render_template(
        "experiment_detail.html",
        student=config.STUDENT_INFO,
        exp=meta,
        metrics=metrics,
        deep_dive=deep_dive,
        prev_exp=prev_exp,
        next_exp=next_exp,
        all_experiments=all_experiments,
        active_page="experiments"
    )
