import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Toxicity Detection Model", layout="centered")

st.title("Toxicity Detection Model")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Enter a comment...")

if user_input:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Preprocess
    processed = user_input.lower()
    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]

    labels = {
        0: "Non-toxic",
        1: "Mild",
        2: "Moderate",
        3: "Severe"
    }

    response = f"Predicted Level: **{labels[prediction]}**"

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

# Sidebar info
st.sidebar.title("Session Statistics")
st.sidebar.write("Total Messages:", len(st.session_state.messages) // 2)