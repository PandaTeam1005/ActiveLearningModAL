from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import preprocess, readDataset
import numpy as np

def termFrecuencyMatrix(X):
    C = CountVectorizer()
    X = C.fit_transform(X)
    # print(C.vocabulary_)
    # D = CountVectorizer(vocabulary=C.vocabulary_)
    # print(D.fit_transform(["Soy cubano americano pero cubano cubano"]))
    # C.get_feature_names()
    return X.toarray()

def tfidfMatrix(X):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(X)
    # tfidf.get_feature_names()
    return X.toarray()

def pmiMatrix(termFrecuency):
    # termFrecuency = np.array(termFrecuency)
    termFrecuency = (termFrecuency.T).dot(termFrecuency)
    total = int(np.sum(termFrecuency))
    # n = len(termFrecuency)
    f_rows = np.sum(termFrecuency,axis=0)
    f_column = np.sum(termFrecuency,axis=1)
    new_matrix = (((termFrecuency * total) / f_rows) / f_column)
    # aux_matrix = np.log2(new_matrix,out=np.zeros_like(termFrecuency),where=(new_matrix>=1))
    return new_matrix

def doc2vecMatrix(X):
    tagged_data = [TaggedDocument(words=_d.split(), tags=[str(i)]) for i, _d in enumerate(X)]
    model = Doc2Vec(vector_size=500)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count,epochs=model.iter,total_words=1000)
    model.save("d2v.model")
    result = []
    for i in range(len(X)):
        result.append(model.docvecs[str(i)])
    return np.array(result)

#X, Y = readDataset('dataset/data.json')
#X_features = termFrecuencyMatrix(X)
#print(X_features.shape)
#save_tfidf = X_features

#X, Y = readDataset('dataset/data.json')
#X_features = doc2vecMatrix(X)
#print(X_features.shape)
# test_data = "yo soy cubano cubano".lower().split()
# v1 = X_features.infer_vector(test_data)
# print("V1_infer", v1.shape)