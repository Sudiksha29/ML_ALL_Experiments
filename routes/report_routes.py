import os
from flask import Blueprint, send_file, abort, jsonify
import config
from services.report_generator import generate_experiment_pdf

report_bp = Blueprint("reports", __name__)

@report_bp.route("/report/<exp_id>/download")
def download_report(exp_id):
    if exp_id not in config.EXPERIMENTS_META:
        abort(404)
    
    try:
        pdf_path = generate_experiment_pdf(exp_id)
        if os.path.exists(pdf_path):
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=f"ML_LabVerse_Report_{exp_id.upper()}_{config.STUDENT_INFO['roll_number']}.pdf",
                mimetype="application/pdf"
            )
        else:
            abort(500, description="PDF generation failed")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
