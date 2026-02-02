import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Take user input
user_input = input("Enter a comment: ")

# Preprocess input (same as training)
cleaned_text = user_input.lower()

# Convert text to numerical vector
vector = vectorizer.transform([cleaned_text])

# Predict severity
prediction = model.predict(vector)[0]

# Map prediction to label
labels = {
    0: "Non-toxic",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

print("Predicted Toxicity Level:", labels[prediction])