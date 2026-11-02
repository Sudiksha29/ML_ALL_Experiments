import os
import json
import joblib
import config

class ModelRegistry:
    _instance = None
    _models = {}
    _metrics = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_all()
        return cls._instance

    def load_all(self):
        """Loads all trained models and metrics into cache."""
        print("[ModelRegistry] Loading models from disk...")
        model_files = {
            "exp02_placement": "exp02_slr_placement.joblib",
            "exp02_hours": "exp02_slr_hours.joblib",
            "exp03_cgpa_iq": "exp03_mlr_cgpa_iq.joblib",
            "exp03_realestate": "exp03_mlr_realestate.joblib",
            "exp04_titanic": "exp04_lr_titanic.joblib",
            "exp04_placement": "exp04_lr_placement.joblib",
            "exp05_student": "exp05_multiclass_student.joblib",
            "exp05_iris": "exp05_multiclass_iris.joblib",
            "exp06_heart": "exp06_dt_heart.joblib",
            "exp07_diabetes": "exp07_knn_diabetes.joblib",
            "exp07_breast_cancer": "exp07_knn_breastcancer.joblib",
            "exp08_banknote": "exp08_gaussian_nb.joblib",
            "exp09_mnb": "exp09_multinomial_nb.joblib",
            "exp09_mnb_222": "exp09_multinomial_nb_222.joblib",
            "exp10_linear_svm": "exp10_linear_svm.joblib",
            "exp11_nonlinear_svm": "exp11_nonlinear_svm.joblib",
            "exp12_rf_diabetes": "exp12_rf_classifier.joblib",
            "exp13_rf_glucose": "exp13_rf_regressor.joblib",
        }

        for key, fname in model_files.items():
            path = os.path.join(config.MODELS_DIR, fname)
            if os.path.exists(path):
                try:
                    loaded = joblib.load(path)
                    self._models[key] = loaded
                    # Register aliases
                    base_key = fname.replace(".joblib", "")
                    self._models[base_key] = loaded
                except Exception as e:
                    print(f"[ModelRegistry] Error loading {key}: {e}")
            else:
                print(f"[ModelRegistry] Warning: {fname} not found.")

        # Load metrics
        if os.path.exists(config.METRICS_FILE):
            try:
                with open(config.METRICS_FILE, "r", encoding="utf-8") as f:
                    self._metrics = json.load(f)
            except Exception as e:
                print(f"[ModelRegistry] Error loading metrics: {e}")
                self._metrics = {}
        else:
            self._metrics = {}

    def get_model(self, key):
        return self._models.get(key)

    def get_metrics(self, exp_id=None):
        if exp_id:
            return self._metrics.get(exp_id, {})
        return self._metrics

    def is_model_available(self, key):
        return key in self._models

registry = ModelRegistry.get_instance()
