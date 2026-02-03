
import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Toxicity Detection Model", layout="centered")

st.title("Toxicity Detection Model - Two User Chat")

# Clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["sender"]):
        st.markdown(f"**{msg['name']}:** {msg['text']}")
        st.markdown(f"Toxicity Level: **{msg['label']}**")

# Select user
sender = st.radio("Select User:", ["User A", "User B"])

# Chat input
user_input = st.chat_input("Type a message...")

if user_input:

    # Convert text to lowercase
    processed = user_input.lower()
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]

    labels = {
        0: "Non-toxic",
        1: "Mild",
        2: "Moderate",
        3: "Severe"
    }

    label = labels[prediction]

    # Determine sender role (for chat style)
    role = "user" if sender == "User A" else "assistant"

    # Save message
    st.session_state.messages.append({
        "sender": role,
        "name": sender,
        "text": user_input,
        "label": label
    })

    # Rerun to display
    st.rerun()