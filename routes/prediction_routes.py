from flask import Blueprint, request, jsonify
from services.prediction_service import predict_experiment

prediction_bp = Blueprint("predictions", __name__)

# Sample input presets for quick testing
SAMPLE_PRESETS = {
    "exp02": {
        "placement": {"cgpa": 8.45},
        "hours": {"hours": 6.5}
    },
    "exp03": {
        "cgpa_iq": {"cgpa": 8.2, "iq": 120.0},
        "realestate": {"house_age": 13.3, "distance": 561.98, "convenience_stores": 5, "latitude": 24.987, "longitude": 121.543}
    },
    "exp04": {
        "titanic": {"pclass": 1, "sex": 1, "age": 29.0, "fare": 85.5},
        "placement": {"cgpa": 8.5, "iq": 125.0}
    },
    "exp05": {
        "student": {"study_hours": 7.0, "attendance": 82.0},
        "iris": {"sepal_length": 6.2, "sepal_width": 2.8, "petal_length": 4.8, "petal_width": 1.8}
    },
    "exp06": {
        "default": {
            "age": 58, "sex": 1, "cp": 2, "trestbps": 135, "chol": 240,
            "fbs": 0, "restecg": 1, "thalach": 142, "exang": 0, "oldpeak": 1.4,
            "slope": 1, "ca": 1, "thal": 2
        }
    },
    "exp07": {
        "diabetes": {
            "Pregnancies": 3, "Glucose": 145, "BloodPressure": 78, "SkinThickness": 28,
            "Insulin": 130, "BMI": 32.4, "DiabetesPedigreeFunction": 0.58, "Age": 42
        },
        "breast_cancer": {
            "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001.0,
            "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001,
            "concave points_mean": 0.1471, "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871
        }
    },
    "exp08": {
        "banknote": {"variance": 2.15, "skewness": 6.32, "curtosis": -0.85, "entropy": -1.24}
    },
    "exp09": {
        "teaching56": {"message": "Please submit your Machine Learning project report before 5 PM tomorrow."},
        "extended": {"message": "Placement aptitude training session starts Saturday at 9 AM in Auditorium."}
    },
    "exp10": {
        "banknote": {"variance": 3.62, "skewness": 8.66, "curtosis": -2.80, "entropy": -0.44}
    },
    "exp11": {
        "banknote": {"variance": -1.45, "skewness": 2.44, "curtosis": 1.88, "entropy": -0.76}
    },
    "exp12": {
        "diabetes": {
            "Pregnancies": 4, "Glucose": 156, "BloodPressure": 82, "SkinThickness": 32,
            "Insulin": 180, "BMI": 35.8, "DiabetesPedigreeFunction": 0.67, "Age": 48
        }
    },
    "exp13": {
        "glucose": {
            "Pregnancies": 2, "BloodPressure": 74, "SkinThickness": 24,
            "Insulin": 115, "BMI": 29.8, "DiabetesPedigreeFunction": 0.45, "Age": 36
        }
    }
}

@prediction_bp.route("/api/predict/<exp_id>", methods=["POST"])
def predict(exp_id):
    if request.is_json:
        data = request.get_json() or {}
        switcher_id = data.get("switcher_id")
        form_data = data.get("inputs", {})
    else:
        form_data = request.form.to_dict()
        switcher_id = form_data.pop("switcher_id", None)

    res = predict_experiment(exp_id, switcher_id, form_data)
    return jsonify(res)

@prediction_bp.route("/api/sample/<exp_id>", methods=["GET"])
def get_sample(exp_id):
    switcher_id = request.args.get("switcher_id")
    presets = SAMPLE_PRESETS.get(exp_id, {})
    if switcher_id and switcher_id in presets:
        return jsonify({"status": "success", "sample": presets[switcher_id]})
    elif "default" in presets:
        return jsonify({"status": "success", "sample": presets["default"]})
    elif presets:
        first_key = list(presets.keys())[0]
        return jsonify({"status": "success", "sample": presets[first_key]})
    return jsonify({"status": "error", "message": "No sample preset available"})
