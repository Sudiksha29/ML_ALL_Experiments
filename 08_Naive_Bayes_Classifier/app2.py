from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your trained model and vectorizer
model = joblib.load("naive_bayes_model2.pkl")
vectorizer = joblib.load("vectorizer2.pkl")

print("=" * 60)
print("CAMPUS MESSAGE DETECTIVE")
print("=" * 60)
print("Model loaded successfully")
print("Vectorizer loaded successfully")
print("Classes:", model.classes_)
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("=" * 60)


# Sample messages
college_messages = [

    {
        "category": "Academic",
        "message": "The DBMS practical examination will be conducted tomorrow in Lab 2."
    },

    {
        "category": "Academic",
        "message": "The internal examination timetable has been published on the student portal."
    },

    {
        "category": "Academic",
        "message": "Students must submit the Machine Learning assignment before Friday."
    },

    {
        "category": "Academic",
        "message": "The faculty has uploaded notes for the next unit."
    },

    {
        "category": "Academic",
        "message": "The project review is scheduled for Monday morning."
    },

    {
        "category": "Placement",
        "message": "The placement cell has announced a campus recruitment drive for eligible students."
    },

    {
        "category": "Placement",
        "message": "Students selected for the internship should report to the training office."
    },

    {
        "category": "Placement",
        "message": "The placement aptitude test is scheduled for Saturday."
    },

    {
        "category": "Placement",
        "message": "Eligible students must register for the upcoming campus recruitment drive."
    },

    {
        "category": "Placement",
        "message": "Bring your resume and identity card for the placement interview."
    },

    {
        "category": "Promotional",
        "message": "Get 50 percent discount on selected products. Shop now and claim your offer."
    },

    {
        "category": "Promotional",
        "message": "Special student discount available. Click to activate."
    },

    {
        "category": "Promotional",
        "message": "Congratulations! You won a free smartphone. Claim it now."
    },

    {
        "category": "Promotional",
        "message": "Flash sale! Grab your favorite products before midnight."
    },

    {
        "category": "Promotional",
        "message": "Exclusive offer! Get a free gift voucher today."
    }
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probabilities = None

    selected_message = ""
    selected_category = ""
    error = None

    if request.method == "POST":

        selected_index = request.form.get(
            "selected_message",
            ""
        )

        custom_message = request.form.get(
            "custom_message",
            ""
        ).strip()

        # Custom message has priority
        if custom_message:

            selected_message = custom_message
            selected_category = "Custom Message"

        # Sample message
        elif selected_index:

            try:

                index = int(selected_index)

                if 0 <= index < len(college_messages):

                    selected_message = (
                        college_messages[index]["message"]
                    )

                    selected_category = (
                        college_messages[index]["category"]
                    )

            except ValueError:

                error = "Invalid message selection."

        else:

            error = "Please select a message or enter your own message."

        # Prediction
        if selected_message and error is None:

            try:

                # Convert text using the SAME vectorizer
                message_vector = vectorizer.transform(
                    [selected_message]
                )

                # Predict class
                prediction = model.predict(
                    message_vector
                )[0]

                # Prediction probabilities
                probability_values = model.predict_proba(
                    message_vector
                )[0]

                probabilities = {}

                for class_name, probability in zip(
                    model.classes_,
                    probability_values
                ):

                    probabilities[class_name] = round(
                        float(probability) * 100,
                        2
                    )

            except Exception as e:

                error = str(e)

    return render_template(
        "index2.html",
        messages=college_messages,
        prediction=prediction,
        probabilities=probabilities,
        selected_message=selected_message,
        selected_category=selected_category,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)