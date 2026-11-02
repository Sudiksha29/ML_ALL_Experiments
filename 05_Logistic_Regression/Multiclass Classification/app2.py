from flask import Flask, render_template, request
import pickle
import pandas as pd

# Create Flask application
app = Flask(__name__)


# Load the trained machine learning model
with open("MCModel2.pkl", "rb") as file:
    model = pickle.load(file)


# Stress level labels
stress_labels = {
    0: "Low Stress",
    1: "Medium Stress",
    2: "High Stress"
}


# Home page
@app.route("/")
def home():
    return render_template("index2.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    
        # Get values from the HTML form
        study_hours = float(request.form["study_hours"])
        sleep_hours = float(request.form["sleep_hours"])
        screen_time = float(request.form["screen_time"])


        # Input validation
        if study_hours < 0 or study_hours > 24:
            return render_template(
                "index2.html",
                prediction_text="Study hours must be between 0 and 24."
            )


        if sleep_hours < 0 or sleep_hours > 24:
            return render_template(
                "index2.html",
                prediction_text="Sleep hours must be between 0 and 24."
            )


        if screen_time < 0 or screen_time > 24:
            return render_template(
                "index2.html",
                prediction_text="Screen time must be between 0 and 24."
            )


        # Create DataFrame with EXACT column names
        # used during model training in Colab
        input_data = pd.DataFrame(
            [[
                study_hours,
                sleep_hours,
                screen_time
            ]],
            columns=[
                "Study_Hours",
                "Sleep_Hours",
                "Screen_Time"
            ]
        )


        # Predict stress level
        prediction = model.predict(input_data)


        # Convert numerical prediction into text
        predicted_class = int(prediction[0])

        result = stress_labels.get(
            predicted_class,
            "Unknown"
        )


        # Display result
        return render_template(
            "index2.html",
            prediction_text=f"Predicted Stress Level: {result}"
        )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)