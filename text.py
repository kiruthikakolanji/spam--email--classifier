#Step 1 - check everthing is ready
import pandas as pd
print("Pandas is working!")

df = pd.read_csv('spam.csv', encoding='latin-1')
print(df.head())         

#Step 2 - Explore and clean the data 
import pandas as pd

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Step 1: Keep only useful columns (v1, v2)
df = df[['v1', 'v2']]

# Step 2: Rename columns to meaningful names
df.columns = ['label', 'message']

# Step 3: Check the data
print(df.head())
print("\n--- Shape of data (rows, columns) ---")
print(df.shape)

print("\n--- Count of spam vs ham ---")
print(df['label'].value_counts())

print("\n--- Check for missing values ---")
print(df.isnull().sum())

#Step 3 - clean the text data
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords list (only needed once)
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()                          # lowercase everything
    text = re.sub(r'[^a-zA-Z]', ' ', text)        # remove numbers/punctuation
    words = text.split()                         # split into individual words
    words = [ps.stem(w) for w in words if w not in stop_words]  # remove stopwords + stem
    return ' '.join(words)                       # join back into a sentence

# Apply to entire dataset
df['clean_message'] = df['message'].apply(clean_text)

# Compare original vs cleaned
print(df[['message', 'clean_message']].head(10))


#Step 4 - convert text to numbers (TF - IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Convert labels to numbers: ham=0, spam=1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Convert text to TF-IDF numbers
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_message']).toarray()
y = df['label_num']

print("Shape of X (features):", X.shape)
print("Shape of y (labels):", y.shape)

# Split into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])



#Step 5 - Train the naive bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create the model
model = MultinomialNB()

# Train the model on training data
model.fit(X_train, y_train)

# Test the model on test data
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Detailed report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Step 6 - save the model
import pickle

# Save the trained model
pickle.dump(model, open('spam_model.pkl', 'wb'))

# Save the vectorizer too (needed to convert new text into numbers later)
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model and vectorizer saved successfully!")