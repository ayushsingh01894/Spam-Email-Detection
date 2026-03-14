from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model/spam_model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    # empty input check
    if message.strip() == "":
        return render_template("index.html", prediction="Please enter a message")

    data = vectorizer.transform([message])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Spam Message"
    else:
        result = "Not Spam"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)