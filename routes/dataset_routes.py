from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import config
from services.dataset_service import get_available_datasets, load_dataset_details, profile_dataframe

dataset_bp = Blueprint("datasets", __name__)

@dataset_bp.route("/datasets")
def dataset_explorer():
    selected_id = request.args.get("id", "placement")
    datasets = get_available_datasets()
    details = load_dataset_details(selected_id)

    return render_template(
        "dataset_explorer.html",
        student=config.STUDENT_INFO,
        datasets=datasets,
        selected_dataset=details,
        active_page="datasets"
    )

@dataset_bp.route("/api/dataset/<dataset_id>")
def get_dataset_api(dataset_id):
    details = load_dataset_details(dataset_id)
    return jsonify(details)

@dataset_bp.route("/api/upload_dataset", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        df = pd.read_csv(file)
        profile = profile_dataframe(df, dataset_id="upload", name=file.filename, filename=file.filename)
        return jsonify(profile)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400
