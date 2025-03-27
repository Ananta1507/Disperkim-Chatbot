import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset chatbot
df = pd.read_csv("chatbot_dataset.csv")

# Konversi teks ke vektor angka
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df["input"])  # Pertanyaan sebagai fitur
y_train = df["response"]  # Jawaban sebagai label

# Training model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Fungsi chatbot untuk menjawab pertanyaan
def chatbot_response(text):
    X_input = vectorizer.transform([text])  # Ubah input user ke format model
    predicted_response = model.predict(X_input)[0]  # Prediksi jawaban
    return predicted_response

# Jalankan chatbot
print("PERKIM A.I: HAI ! Saya Disperkim BOT ada yang bisa saya bantu ?.")
while True:
    user_input = input("Kamu: ")
    if user_input.lower() == "bye":
        print("Chatbot: Sampai jumpa!")
        break
    try:
        print("Chatbot:", chatbot_response(user_input))
    except:
        print("Chatbot: Maaf, saya tidak mengerti.")

