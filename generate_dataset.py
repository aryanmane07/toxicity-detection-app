import random
import pandas as pd

# Word banks
neutral_phrases = [
    "I really like this idea",
    "Thank you for your help",
    "This is a useful suggestion",
    "I appreciate your feedback",
    "That sounds like a good plan",
    "Great explanation, very clear",
    "I enjoyed reading this",
    "This information is helpful",
    "Nice work on this project",
    "Well written and informative"
]

mild_phrases = [
    "That was a silly mistake",
    "You are being annoying",
    "This is kind of stupid",
    "Stop acting weird",
    "That comment makes no sense",
    "You are not very smart",
    "This is pointless",
    "You sound confused",
    "That was unnecessary",
    "Please stop being rude"
]

moderate_phrases = [
    "You are completely useless",
    "This is absolute garbage",
    "You have no idea what you are doing",
    "That was incredibly dumb",
    "You are embarrassingly ignorant",
    "This is pathetic nonsense",
    "Nobody respects your opinion",
    "You always ruin everything",
    "That was extremely stupid",
    "You are a total failure"
]

severe_phrases = [
    "I will hurt you",
    "You deserve to suffer",
    "I will destroy everything you care about",
    "People like you should disappear",
    "I will make you regret this",
    "You are a worthless human being",
    "I will find you",
    "You deserve punishment",
    "I will ruin your life",
    "You should be banned forever"
]

def generate_samples(phrases, severity, count):
    data = []
    for _ in range(count):
        sentence = random.choice(phrases)
        data.append([sentence, severity])
    return data

data = []
data += generate_samples(neutral_phrases, 0, 1000)
data += generate_samples(mild_phrases, 1, 1000)
data += generate_samples(moderate_phrases, 2, 1000)
data += generate_samples(severe_phrases, 3, 1000)
import pandas as pd

df = pd.DataFrame(data, columns=["comment_text", "severity"])
df.to_csv("custom_dataset.csv", index=False)

print("Custom dataset created successfully!")


random.shuffle(data)

df = pd.DataFrame(data, columns=["comment_text", "severity"])
df.to_csv("custom_train.csv", index=False)

print("Custom dataset generated successfully!")