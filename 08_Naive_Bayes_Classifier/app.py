from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained models
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("naive_bayes_model.pkl")

# Sample college messages
messages = [
    {
        "category": "Workshop",
        "message": "A hands-on Artificial Intelligence workshop will be conducted in the Computer Lab on Friday at 10 AM."
    },
    {
        "category": "Exam",
        "message": "The internal examination timetable has been released. Students must check the schedule on the college portal."
    },
    {
        "category": "Placement",
        "message": "The placement cell has announced a campus recruitment drive. Eligible students must register before Wednesday."
    },
    {
        "category": "Scholarship",
        "message": "Students eligible for the government scholarship must submit the required documents to the college office before the deadline."
    },
    {
        "category": "College Event",
        "message": "The annual technical festival will be held next week. Students interested in participating should register with their department."
    },
    {
        "category": "Assignment",
        "message": "Students must submit the Machine Learning assignment on the LMS before Friday evening."
    },
    {
        "category": "Practical",
        "message": "The DBMS practical examination will be conducted in Lab 2 tomorrow. Students should bring their lab manual."
    },
    {
        "category": "Lecture",
        "message": "Today's Data Science lecture has been shifted to Room 305 and will begin at 11 AM."
    },
    {
        "category": "Internship",
        "message": "Students selected for the summer internship must report to the training and placement office tomorrow."
    },
    {
        "category": "Coding Competition",
        "message": "Registration is open for the college coding competition. Interested students should register before the closing date."
    },
    {
        "category": "Shopping Offer",
        "message": "Exclusive student offer! Get 60 percent off on selected products. Shop now and claim your discount."
    },
    {
        "category": "Prize Promotion",
        "message": "Congratulations! You have been selected for a special reward. Click the link now to claim your free gift."
    }
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probabilities = None
    selected_message = ""
    selected_category = ""

    if request.method == "POST":

        # Get values from HTML
        selected_index = request.form.get("selected_message")
        custom_message = request.form.get("custom_message", "").strip()

        # Custom message has priority
        if custom_message:

            selected_message = custom_message
            selected_category = "Custom Message"

        # Otherwise use selected sample message
        elif selected_index:

            try:
                index = int(selected_index)

                if 0 <= index < len(messages):

                    selected_message = messages[index]["message"]
                    selected_category = messages[index]["category"]

            except ValueError:
                pass

        # Make prediction
        if selected_message:

            # Convert text into features
            text_vector = vectorizer.transform(
                [selected_message]
            )

            # Predict class
            prediction = model.predict(
                text_vector
            )[0]

            # Get probabilities
            probability_values = model.predict_proba(
                text_vector
            )[0]

            # Create probability dictionary
            probabilities = {}

            for class_name, probability in zip(
                model.classes_,
                probability_values
            ):

                probabilities[class_name] = round(
                    probability * 100,
                    2
                )

    return render_template(
        "index.html",
        messages=messages,
        prediction=prediction,
        probabilities=probabilities,
        selected_message=selected_message,
        selected_category=selected_category
    )


if __name__ == "__main__":
    app.run(debug=True)