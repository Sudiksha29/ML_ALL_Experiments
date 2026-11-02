import os
from flask import Flask, render_template
import config
from services.model_registry import registry
from services.history_service import init_history_db

from routes.dashboard_routes import dashboard_bp
from routes.experiment_routes import experiment_bp
from routes.prediction_routes import prediction_bp
from routes.dataset_routes import dataset_bp
from routes.comparison_routes import comparison_bp
from routes.learning_routes import learning_bp
from routes.report_routes import report_bp

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "ml-labverse-secret-key-ingale-2026"

    # Initialize SQLite database
    init_history_db()

    # Pre-warm model registry
    registry.load_all()

    # Register Blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(experiment_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(comparison_bp)
    app.register_blueprint(learning_bp)
    app.register_blueprint(report_bp)

    @app.context_processor
    def inject_global_data():
        return {
            "student_info": config.STUDENT_INFO,
            "all_experiments": config.EXPERIMENTS_META
        }

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def internal_error(e):
        return render_template("500.html", error=e), 500

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="127.0.0.1")
