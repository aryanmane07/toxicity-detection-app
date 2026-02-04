from flask import Flask, render_template, request, redirect
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

labels = {
    0: "Non-toxic",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

messages = []

def predict_toxicity(text):
    text_lower = text.lower()
    text_vector = vectorizer.transform([text_lower])
    prediction = model.predict(text_vector)[0]
    return labels[prediction]


@app.route("/userA", methods=["GET", "POST"])
def userA():
    if request.method == "POST":
        text = request.form["message"]
        severity = predict_toxicity(text)

        messages.append({
            "sender": "User A",
            "text": text,
            "severity": severity
        })

        return redirect("/userA")

    return render_template(
        "chat.html",
        messages=messages,
        current_user="User A",
        total=len(messages)
    )


@app.route("/userB", methods=["GET", "POST"])
def userB():
    if request.method == "POST":
        text = request.form["message"]
        severity = predict_toxicity(text)

        messages.append({
            "sender": "User B",
            "text": text,
            "severity": severity
        })

        return redirect("/userB")

    return render_template(
        "chat.html",
        messages=messages,
        current_user="User B",
        total=len(messages)
    )


@app.route("/clear")
def clear_chat():
    global messages
    messages = []
    return redirect("/userA")


