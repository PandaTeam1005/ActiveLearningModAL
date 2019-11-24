from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json
import numpy as np

NAIVE_BAYES_CLASSIFIER = MultinomialNB()
SVM_CLASSIFIER = SVC(kernel='linear')
KNN_CLASSIFIER = KNeighborsClassifier(n_neighbors=4, metric='euclidean')

def classify(classifier, L, U):
    X_train = L.element()
    Y_train = [L.item_class(e) for e in X_train]
    X_test = U.element()

    classifier.fit(X_train, Y_train)
    Y_predict = classifier.predict(X_test)

    prob_list = classifier.predict_proba(X_test)
    high_prop = [max(l) for l in prob_list]
    result = zip(X_test,Y_predict,high_prop)

    # return [(item, class, prob)]
    return result


# X_comment = []
# Y_comment = []
# words = set()
# for obj in comments:
#     text = obj['text'].split()
#     for w in text:
#         words.add(w)
#     X_comment.append(text)
#     Y_comment.append(obj['answers'][0]['answer'])

# n = len(X_comment)
# m = len(words)
# term_document = np.zeros((n,m))
# index_comment = {(' '.join(name)): i for i,name in enumerate(X_comment)}
# index_word = {name: i for i,name in enumerate(words)}

# for comment in X_comment:
#     j = index_comment[' '.join(comment)]
#     for w in comment:
#         i = index_word[w]
#         term_document[j,i] += 1

# print(term_document.shape)

# X_train, X_test, Y_train, Y_test = train_test_split(term_document, Y_comment,test_size=0.2)

# print(X_train.shape)

# classifier = MultinomialNB()
# classifier.fit(X_train, Y_train)
# Y_predict = classifier.predict(X_test)
# # print(Y_predict[0])
# # print(X_test[0])
# # print(classifier.predict_proba(X_test))

# print(classification_report(Y_test, Y_predict))
# print(confusion_matrix(Y_test, Y_predict))

# # classifier.fit(X_train, Y_train)
# # classifier.predict([X])
# # classifier.predict_proba(X)

# classification_report(Y_test, Y_predict)
# confusion_matrix(Y_test, Y_predict)








