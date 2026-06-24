from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and vectorizer
model = pickle.load(
    open(os.path.join(BASE_DIR, "model", "spam_model.pkl"), "rb")
)

vectorizer = pickle.load(
    open(os.path.join(BASE_DIR, "model", "vectorizer.pkl"), "rb")
)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    # Empty input check
    if message.strip() == "":
        return render_template(
            "index.html",
            prediction="Please enter a message"
        )

    data = vectorizer.transform([message])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Spam Message"
    else:
        result = "Not Spam"

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)