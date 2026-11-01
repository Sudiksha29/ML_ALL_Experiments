from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained Logistic Regression model
model = pickle.load(
    open("BCModel2.pkl", "rb")
)


@app.route("/")
def home():
    return render_template("index2.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get Account ID
    account_id = request.form["account_id"]

    # Get Account Type
    account_type = request.form["account_type"]

    # Create input DataFrame
    new_account = pd.DataFrame({
        "Account_type": [account_type]
    })

    # Make prediction
    prediction = model.predict(new_account)

    # Convert prediction to card name
    if prediction[0] == 0:
        card = "Diamond"
    else:
        card = "Silver"

    return render_template(
        "index2.html",
        account_id=account_id,
        account_type=account_type,
        prediction_text=f"Predicted Card Assigned: {card}"
    )


if __name__ == "__main__":
    app.run(debug=True)