from flask import Flask, render_template, request
import pickle

# Create Flask application

app = Flask(__name__)

# Load the trained machine learning model

with open("MCModel.pkl", "rb") as file:
    model = pickle.load(file)

# Home Page

@app.route("/")
def home():
    return render_template("index.html")

# Prediction Page

@app.route("/predict", methods=["POST"])
def predict():
    
        # Get input values from HTML form
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])

        # Validate study hours
        if study_hours < 0:
            return render_template(
                "index.html",
                prediction_text="Study hours cannot be negative."
            )

        # Validate attendance
        if attendance < 0 or attendance > 100:
            return render_template(
                "index.html",
                prediction_text="Attendance must be between 0 and 100."
            )

        # Prepare data for prediction
        input_data = [[study_hours, attendance]]

        # Make prediction
        prediction = model.predict(input_data)

        # Convert prediction number into performance category
        performance_labels = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        # Get final result
        result = performance_labels.get(
            int(prediction[0]),
            "Unknown Performance"
        )

        # Display result on website
        return render_template(
            "index.html",
            prediction_text=f"Predicted Performance Level: {result}"
        )


# Run Flask Application

if __name__ == "__main__":
    app.run(debug=True)
