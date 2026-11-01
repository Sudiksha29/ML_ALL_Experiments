from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained Decision Tree model
with open("DTModel.pkl", "rb") as file:
    model = pickle.load(file)


# Feature order must match model training
FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get JSON data from frontend
        data = request.get_json(force=True)

        print("Received data:", data)

        # Check whether data is received
        if not data:
            return jsonify({
                "error": "No input data received."
            }), 400


        # Check missing fields
        missing_fields = []

        for feature in FEATURE_ORDER:
            if feature not in data or data[feature] == "":
                missing_fields.append(feature)


        if missing_fields:
            return jsonify({
                "error": "Missing values for: " + ", ".join(missing_fields)
            }), 400


        # Convert input values to float
        features = []

        for feature in FEATURE_ORDER:
            features.append(float(data[feature]))


        # Convert to NumPy array
        input_data = np.array(features).reshape(1, -1)

        print("Input array:", input_data)


        # Prediction
        prediction = model.predict(input_data)[0]

        print("Prediction:", prediction)


        # Prediction probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            confidence = round(
                float(np.max(probabilities)) * 100,
                2
            )

        else:

            confidence = 0


        # Result
        if int(prediction) == 1:

            label = "Heart Disease Detected"

        else:

            label = "No Heart Disease Detected"


        return jsonify({

            "prediction": int(prediction),

            "label": label,

            "confidence": confidence

        })


    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({

            "error": str(e)

        }), 500


if __name__ == "__main__":
    app.run(debug=True)