from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Severity labels
labels = {
    0: "Non-toxic",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

# Store chat messages
messages = []

@app.route("/userA")
def userA():
    return render_template("chat.html", username="User A")

@app.route("/userB")
def userB():
    return render_template("chat.html", username="User B")

@socketio.on("send_message")
def handle_message(data):
    sender = data["sender"]
    text = data["text"]

    text_lower = text.lower()
    text_vector = vectorizer.transform([text_lower])
    prediction = model.predict(text_vector)[0]
    severity = labels[prediction]

    message_data = {
        "sender": sender,
        "text": text,
        "severity": severity
    }

    messages.append(message_data)

    emit("receive_message", message_data, broadcast=True)

if __name__ == "__main__":
    socketio.run(app, debug=True)