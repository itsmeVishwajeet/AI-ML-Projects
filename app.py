from flask import Flask, render_template, request
from keras.models import load_model
import pickle
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the sentiment analysis model
model = load_model('best_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_text(text):
    '''Function to preprocess the text'''
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Apply any other necessary preprocessing steps
    
    return text

def predict_sentiment(text):
    '''Function to predict the sentiment class of the text'''
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len = 50

    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Convert the preprocessed text to a sequence of integers using the tokenizer
    text_sequence = tokenizer.texts_to_sequences([preprocessed_text])

    # Pad the sequence to the same length as the training data
    padded_sequence = pad_sequences(text_sequence, padding='post', maxlen=max_len)

    # Do the prediction using the loaded model
    predicted_class = model.predict(padded_sequence).argmax(axis=1)

    # Return the predicted sentiment class
    return sentiment_classes[predicted_class[0]]

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/sentiment')
def sentiment():
    return render_template('index.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']
    sentiment = predict_sentiment(sentence)
    return render_template('index.html', sentence=sentence, sentiment=sentiment)

if __name__ == '__main__':
    app.run(port=8090, debug=True)

