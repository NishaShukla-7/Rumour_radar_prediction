from flask import Flask, request, render_template
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news']
    clean = preprocess(input_text)
    vect_text = vectorizer.transform([clean]).toarray()
    prediction = model.predict(vect_text)[0]
    result = "REAL" if prediction == 1 else "RUMOUR / FAKE"
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)
