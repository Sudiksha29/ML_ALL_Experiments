from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "linear_svm_banknote_model.pkl"

# This is the trained Pipeline from the Colab notebook. It includes
# StandardScaler and the Linear SVM classifier.
model = joblib.load(MODEL_PATH)

FEATURES = ["variance", "skewness", "curtosis", "entropy"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = {feature: "" for feature in FEATURES}

    if request.method == "POST":
        try:
            row = {}
            for feature in FEATURES:
                raw_value = request.form.get(feature, "").strip()
                values[feature] = raw_value
                if not raw_value:
                    raise ValueError(f"Please enter a value for {feature.title()}.")
                row[feature] = float(raw_value)

            # Keep the same feature names and order used to train the model.
            input_data = pd.DataFrame([row], columns=FEATURES)
            predicted_class = int(model.predict(input_data)[0])

            if predicted_class == 0:
                prediction = {
                    "label": "Genuine Banknote",
                    "class": 0,
                    "description": "The trained Linear SVM model classified this sample as genuine.",
                    "kind": "genuine",
                }
            else:
                prediction = {
                    "label": "Counterfeit Banknote",
                    "class": 1,
                    "description": "The trained Linear SVM model classified this sample as counterfeit.",
                    "kind": "counterfeit",
                }

        except ValueError as exc:
            error = str(exc) if str(exc).startswith("Please enter") else (
                "Please enter valid numeric values in all four fields."
            )
        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction could not be completed. Check that the model file and dependencies are correct."

    return render_template("index.html", prediction=prediction, error=error, values=values)


if __name__ == "__main__":
    # Run locally for the practical. Debug mode is enabled for development only.
    app.run(debug=True)
