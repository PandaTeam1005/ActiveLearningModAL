from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import json
import numpy as np
from utils import readDataset
from vectorize import doc2vecMatrix

NAIVE_BAYES_CLASSIFIER = MultinomialNB()
SVM_CLASSIFIER = SVC(kernel="rbf",probability=True,gamma="auto")
KNN_CLASSIFIER = KNeighborsClassifier(n_neighbors=20, metric='euclidean')

X, Y = readDataset('dataset/data.json')
pos = []
neg = []
neu = []
obj = []

for i,y in enumerate(Y):
    if y == "Objetivo":
        obj.append(X[i])
    elif y == "Neutro":
        neu.append(X[i])
    elif y == "Negativo":
        neg.append(X[i])
    else:
        pos.append(X[i])

print([len(pos),len(neg),len(neu),len(obj)])
X = pos + neg
Y = ["Positivo"]*len(pos) + ["Negativo"]*len(neg)

X_features = doc2vecMatrix(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y,test_size=0.2)

print("X_train_shape", X_train.shape)
# print(Y_train)
print("Negativo","Neutro","Objetivo","Positivo")
r = [0]*4
for y in Y_train:
    if y == "Objetivo":
        r[2] += 1
    elif y == "Neutro":
        r[1] += 1
    elif y == "Negativo":
        r[0] += 1
    else:
        r[3] += 1
print(r)

classifier = KNN_CLASSIFIER
classifier.fit(X_train, Y_train)
Y_predict = classifier.predict(X_test)
# print(Y_predict[0])
# # print(X_test[0])
# # print(classifier.predict_proba(X_test))

r = [0]*4
for y in Y_test:
    if y == "Objetivo":
        r[2] += 1
    elif y == "Neutro":
        r[1] += 1
    elif y == "Negativo":
        r[0] += 1
    else:
        r[3] += 1
print("Test", r)

r = [0]*4
for y in Y_predict:
    if y == "Objetivo":
        r[2] += 1
    elif y == "Neutro":
        r[1] += 1
    elif y == "Negativo":
        r[0] += 1
    else:
        r[3] += 1
print("Predict", r)

print(classification_report(Y_test, Y_predict))
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict))
# # classifier.fit(X_train, Y_train)
# # classifier.predict([X])
# # classifier.predict_proba(X)

# classification_report(Y_test, Y_predict)
# confusion_matrix(Y_test, Y_predict)








