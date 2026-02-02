import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("custom_dataset.csv")
print("Severity distribution:")
print(df['severity'].value_counts())

# Create severity labels

df['comment_text'] = df['comment_text'].str.lower()

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['comment_text'])
y = df['severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")
