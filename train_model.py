import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load dataset (Make sure the CSV file is in the same folder)
df = pd.read_csv('fake_or_real_news.csv')  # Replace with your dataset if needed
df = df[['text', 'label']]
df['clean'] = df['text'].apply(preprocess)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean']).toarray()
y = df['label'].apply(lambda x: 1 if x == 'real' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ Model and vectorizer saved successfully!")
