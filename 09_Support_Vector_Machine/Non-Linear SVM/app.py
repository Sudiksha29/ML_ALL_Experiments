from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "nonlinear_svm_banknote_model.pkl"

FEATURES = ["variance", "skewness", "curtosis", "entropy"]
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    values = {feature: "" for feature in FEATURES}

    if request.method == "POST":
        try:
            data = {}
            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()
                values[feature] = raw
                if not raw:
                    raise ValueError("Please fill in all four feature fields.")
                data[feature] = float(raw)

            sample = pd.DataFrame([data], columns=FEATURES)
            predicted_class = int(model.predict(sample)[0])

            result = {
                "class": predicted_class,
                "label": "GENUINE BANKNOTE" if predicted_class == 0 else "COUNTERFEIT BANKNOTE",
                "detail": (
                    "The RBF SVM classified this sample as genuine."
                    if predicted_class == 0
                    else "The RBF SVM classified this sample as counterfeit."
                ),
                "kind": "genuine" if predicted_class == 0 else "counterfeit",
            }
        except ValueError as exc:
            error = str(exc) if str(exc).startswith("Please fill") else "Enter valid numeric values in all fields."
        except Exception:
            app.logger.exception("Prediction error")
            error = "Prediction failed. Confirm the model file and installed dependencies."

    return render_template("index.html", result=result, error=error, values=values)


if __name__ == "__main__":
    app.run(debug=True)
