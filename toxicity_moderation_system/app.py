from flask import Flask, render_template, request, redirect, session
from flask_socketio import SocketIO, emit
from datetime import datetime
import pickle
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# IMPORTANT: specify async_mode
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# In-memory storage
messages = []

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        session["username"] = username
        return redirect("/chat")
    return render_template("login.html")


@app.route("/chat")
def chat():
    if "username" not in session:
        return redirect("/login")
    return render_template("chat.html", username=session["username"])


@app.route("/admin")
def admin():
    return render_template("admin.html", messages=messages)


# ==============================
# SOCKET EVENTS
# ==============================

@socketio.on("send_message")
def handle_message(data):
    text = data["text"]
    username = data["username"]

    prediction = model.predict(vectorizer.transform([text]))[0]

    timestamp = datetime.now().strftime("%H:%M")

    message_data = {
        "username": username,
        "text": text,
        "severity": str(prediction),
        "timestamp": timestamp
    }

    messages.append(message_data)

    emit("receive_message", message_data, broadcast=True)


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))