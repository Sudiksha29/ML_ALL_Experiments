from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Multiple Linear Regression model
model = pickle.load(open("MLRModel2.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index2.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Read inputs from the HTML form
    experience = float(request.form['experience'])
    education = float(request.form['education'])

    # Predict monthly salary
    prediction = model.predict(np.array([[experience, education]]))

    return render_template(
        "index2.html",
        prediction_text=f"Predicted Monthly Salary : {prediction[0]:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)
