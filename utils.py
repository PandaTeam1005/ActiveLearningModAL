from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import json

def readDataset(filename):
    with open(filename, encoding='utf-8') as data:
        comments = json.load(data)

    X_comment = []
    Y_comment = []
    words = set()
    for obj in comments:
        text = obj['text'].split()
        for w in text:
            words.add(w)
        X_comment.append(preprocess(obj['text']))
        Y_comment.append(obj['answers'][0]['answer'])
    return X_comment, Y_comment

def preprocess(text):
    result_text = set()
    stop_words = set(stopwords.words('spanish'))
    sb = SnowballStemmer('spanish')
    for w in text.split():
        if not (w in stop_words or w in punctuation or w.isdigit()):
            result_text.add(sb.stem(w))
    return ' '.join(list(result_text))
