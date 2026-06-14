#Step 7 - Build the Streamlit web app 

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (needed here too, since this is a separate file)
nltk.download('stopwords')

# Load saved model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Same cleaning function as before
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ---- Web App UI ----
st.title("📧 Spam Email Classifier")
st.write("Enter a message below to check if it's Spam or Not!")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT Spam (Ham)")
    else:
        st.warning("Please enter a message first!")