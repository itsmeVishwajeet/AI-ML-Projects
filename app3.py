from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model
model = load_model('best_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

import re

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = text.strip()
    return text

# Preprocess the text and perform sentiment analysis
def preprocess_and_predict(text):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    sequence = pad_sequences(sequence, maxlen=50)  # Adjust maxlen based on your tokenizer configuration
    prediction = model.predict(sequence).argmax(axis=1)[0]
    
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    sentiment = sentiment_classes[prediction]
    
    return sentiment

# Define the routes
@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['sentence']
    sentiment = preprocess_and_predict(text)
    return render_template('index2.html', sentence=text, sentiment=sentiment)

# Run the Flask application
if __name__ == '__main__':
    app.run(port=8080)  # Change the port number as needed

