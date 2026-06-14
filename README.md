spam--email--classifier

AI-powered spam email classifier using ML

📧 Spam Email Classifier

A machine learning web app that classifies messages as Spam or Not Spam (Ham) using Natural Language Processing and Naive Bayes algorithm.

🚀 Live Demo
[Add your deployed app link here once deployed]

📋 Features
- Real-time spam detection
- Clean and simple web interface
- 98% model accuracy

🛠️ Tech Stack
- Python– Core programming language
- Pandas– Data manipulation
- NLTK (Natural Language Toolkit) – Text preprocessing (stopword removal, stemming)
- Scikit-learn– TF-IDF vectorization & Naive Bayes model
- Streamlit– Web app interface

📊 Dataset
[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) – 5,572 labeled messages (spam/ham)

🧠 How It Works
1. Text is cleaned (lowercase, remove punctuation/numbers, remove stopwords, stemming)
2. Cleaned text is converted to numerical features using TF-IDF
3. A Multinomial Naive Bayes model predicts spam or ham
4. Results are displayed instantly on the web app

📈 Model Performance
- Accuracy: 98%
- Precision (Spam): 100%
- Recall (Spam): 82%

💻 How to Run Locally

1. Clone this repository
git clone https://github.com/yourusername/spam-email-classifier.git
cd spam-email-classifier

2. Install dependencies
pip install -r requirements.txt

3. Run the app
python -m streamlit run app.py


📁 Project Structure
spam-email-classifier/

1. app.py            - Streamlit web app
2. text.py           - Model training script
3. spam_model.pkl    -Trained model
4. vectorizer.pkl    -TF-IDF vectorizer
5. spam.csv          -Dataset
6. requirements.txt  -Dependencies
7.README.md          -Project documentation
