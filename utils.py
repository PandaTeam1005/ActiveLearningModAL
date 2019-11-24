from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
from unicodedata import normalize
import json
import re

def unicodeText(dictionary):
    keys = list(dictionary.keys())
    for k in keys:
        original_k = k
        k = re.sub(
            r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
            normalize( "NFD", k), 0, re.I
        )
        k = normalize( 'NFC', k)
        if original_k != k:
            dictionary[k] = original_k
            # print(original_k,"->",k)

def readDataset(filename):
    with open(filename, encoding='utf-8') as data:
        comments = json.load(data)
    # with open("dataset/word_dict_es.json", encoding='utf-8') as data:
    #     dictionary = json.load(data)
    # unicodeText(dictionary)
    dictionary = {}
    X_comment = []
    Y_comment = []
    words = set()
    for obj in comments:
        # text = word_tokenize(obj['text'])
        text = preprocess(obj['text'],dictionary).split()
        for w in text:
            try:
                dictionary[w]
            except Exception:
                words.add(w)
        X_comment.append(" ".join(text))
        Y_comment.append(obj['answers'][0]['answer'])
    # print(words)
    # print(len(words))
    return X_comment, Y_comment

def preprocess(text, dicc = {}):
    result_text = set()
    text = text.lower()
    text = " ".join(word_tokenize(text))
    regex = re.compile('[%s]' % re.escape(punctuation))
    text = regex.sub('',text)
    stop_words = set(stopwords.words('spanish'))
    sb = SnowballStemmer('spanish')
    # new_list = replaceUnicode(text.split(), dicc)
    for w in text.split():
        if not (w in stop_words or w in punctuation or w.isdigit()):            
            result_text.add(w)
    return ' '.join(list(result_text))

def replaceUnicode(parameter_list, dicc):
    new_list = []
    for w in parameter_list:
        try:
            dicc[w]
            if type(dicc[w])==str:
                w = dicc[w]
            new_list.append(w)
        except Exception:
            new_list.append(w)
    return new_list