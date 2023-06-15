import nltk
import re
from urllib.parse import urlparse
from spacy import load
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('omw-1.4') # Open Multilingual Wordnet, this is an lexical database
# nltk.download('wordnet')
# nltk.download('wordnet2022')
# nltk.download('punkt')
nltk.download('stopwords')


from flask import Flask, request, jsonify
import pickle
import warnings

app = Flask(__name__)
warnings.filterwarnings('ignore')

# Load the pickled predictor function and tfidf object
with open('python backend/api/model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('python backend/api/tfidf.pkl', 'rb') as file:
    tf = pickle.load(file)

lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))

def textProcess(sent):
    try:
        # brackets replacing by space
        sent = re.sub('[][)(]',' ',sent)

        # url removing
        sent = [word for word in sent.split() if not urlparse(word).scheme]
        sent = ' '.join(sent)

        # removing escap characters
        sent = re.sub(r'\@\w+','',sent)

        # removing html tags
        sent = re.sub(re.compile("<.*?>"),'',sent)

        # getting only characters and numbers from text
        sent = re.sub("[^A-Za-z0-9]",' ',sent)

        # lower case all words
        sent = sent.lower()

        # strip all words from sentences
        sent = [word.strip() for word in sent.split()]
        sent = ' '.join(sent)

        # word tokenization
        tokens = word_tokenize(sent)

        # removing words which are in stopwords
        for word in tokens:
            if word in stop_words:
                tokens.remove(word)

        # lemmatization
        sent = [lemmatizer.lemmatize(word) for word in tokens]
        sent = ' '.join(sent)
        return sent

    except Exception as ex:
        print(sent,"\n")
        print("Error ",ex)




@app.route('/')
def home():
    return "Go to /predict for Stress Prediction"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assuming the request data is in JSON format
    text = data['text']  # Assuming the input text is provided as 'text' field in the JSON data
    
    processed = textProcess(text)
    embedded_words = tf.transform([processed])
    res = model.predict(embedded_words)
    if res[0] == 1:
        prediction = 1 # this person is in stress
    else:
        prediction = 0 #this person is not in stress

    response = {'prediction': prediction}  # Create a response dictionary with the prediction result
    return jsonify(response)  # Return the response as JSON

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask application
