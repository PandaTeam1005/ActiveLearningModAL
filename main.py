from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from modAL.models import ActiveLearner,Committee
from sklearn.svm import SVC
import numpy as np
import random
from utils import readDataset
from vectorize import *
from sklearn.naive_bayes import GaussianNB
from copy import deepcopy
from sklearn.metrics import accuracy_score

#'Objetivo':2,'Negativo':0,'Positivo':3,"Neutro":2
#Negativo V Positivo :1
class Committee2Level:
    def __init__(self,estimators_1,estimators_2,X,Y):
        indexs2 = list(filter(lambda i: Y[i]!=2,range(len(Y))))
        X1 = X
        Y1 = np.array([i if i==2 else 1 for i in Y])
        X2 = X[indexs2]
        Y2 = Y[indexs2]
        learners1 = [ActiveLearner(e,X_training=X1,y_training=Y1) for e in estimators_1]
        learners2 = [ActiveLearner(e,X_training=X2,y_training=Y2) for e in estimators_2]
        self.osCommittee = Committee(learners1)
        self.pnCommittee = Committee(learners2)
        self.choice_indx = 0
    def teach(self,X,Y):
        indexs2 = list(filter(lambda i: Y[i]!=2,range(len(Y))))
        X1 = X
        Y1 = np.array([i if i==2 else 1 for i in Y])
        X2 = X[indexs2]
        Y2 = Y[indexs2]
        self.osCommittee.teach(X1,Y1)
        if len(Y2) > 0:
            self.pnCommittee.teach(X2,Y2)
    def predict(self,X):
        predicts1 = self.osCommittee.predict(X)
        indexs = range(len(X))
        predicts2 = self.pnCommittee.predict(X)
        predicts = [predicts1[i] if predicts1[i]==2 else predicts2[i] for i in indexs]
        return predicts
    def query(self,X,Y):
        indexs2 = list(filter(lambda i: Y[i]!=2,range(len(Y))))
        X1 = X
        X2 = X[indexs2]
        q2 = ([],2)
        if len(X2)>0:
            q2 = self.pnCommittee.query(X2)
        alternatives = [self.osCommittee.query(X1), q2]
        r = alternatives[self.choice_indx%2]
        #print(self.choice_indx)
        self.choice_indx +=1
        if len(r[0]) == 0:
            return alternatives[self.choice_indx%2]
        return r
           
def cmte_loop(estimator1,estimator2,X_0,Y_0,X_train,Y_train,X_test,Y_test,indexs,n=5):
    #learners = []
    X_pool = deepcopy(np.delete(X_train,indexs,axis=0))
    Y_pool = deepcopy(np.delete(Y_train,indexs))
    #committee = Committee2Level(estimator1,estimator2,X_0,Y_0)
    committee = Committee([ActiveLearner(e,X_training=X_0,y_training=Y_0) for e in estimator1])
    accuracies = []
    while len(X_pool)>0:
        #query_indxs,_ = committee.query(X_pool,Y_pool)
        query_indxs,_ = committee.query(X_pool) 
        committee.teach(X_pool[query_indxs],Y_pool[query_indxs])
        
        X_0 = np.append(X_0,X_pool[query_indxs],axis=0)
        Y_0 = np.append(Y_0,Y_pool[query_indxs][0])
        X_pool = np.delete(X_pool,query_indxs,axis=0)
        Y_pool = np.delete(Y_pool,query_indxs,axis=0)
        accuracies.append(evaluate(committee,X_0,Y_0,X_test,Y_test))
    return (committee,accuracies)
def evaluate(commitee,X,Y,X_test,Y_test):
    e = SVC(kernel="linear",gamma=0.001)
    cmt_p = commitee.predict(X)
    cmt_pt = commitee.predict(X_test)
    e.fit(X,Y)
    sv_p = e.predict(X)
    sv_pt = e.predict(X_test)
    al_tr = accuracy_score(Y,cmt_p)
    al_te = accuracy_score(Y_test,cmt_pt)
    sv_tr = accuracy_score(Y,sv_p)
    sv_te = accuracy_score(Y_test,sv_pt)
    return (al_tr,al_te,sv_tr,sv_te)



def Query_by_committee(X,Y,estimator1,estimator2):
    X = deepcopy(X)
    Y = deepcopy(Y)    
    indexs = np.random.choice(range(len(X)),size=50,replace=False)
    X_test = X[indexs]
    Y_test = Y[indexs]
    X_train = np.delete(X,indexs,axis=0)
    Y_train = np.delete(Y,indexs)
    indexs = np.random.choice(range(len(X_train)),size=50,replace=False)
    X_0 = X[indexs]
    Y_0 = Y[indexs]
    _,accuracies = cmte_loop(estimator1,estimator2,X_0,Y_0,X_train,Y_train,X_test,Y_test,indexs)
    return accuracies

def N_iter(X,Y,n=30):
    trains_al = []
    tests_al = []
    trains_svc = []
    test_svc = []
    

    e1 = SVC(kernel="rbf",probability=True,gamma=0.001)
    e2 = SVC(kernel="linear",probability=True,gamma=0.001)
    e3 = KNeighborsClassifier(n_neighbors=10)
    e4 = SVC(kernel="rbf",probability=True,gamma=0.001)
    e5 = SVC(kernel="rbf",probability=True,gamma="auto")

    
    e11 = SVC(kernel="rbf",probability=True,gamma=0.001)
    e12 = SVC(kernel="linear",probability=True,gamma=0.001)
    e13 = KNeighborsClassifier(n_neighbors=10)
    e14 = SVC(kernel="rbf",probability=True,gamma=0.001)
    e15 = SVC(kernel="rbf",probability=True,gamma="auto")



    for i in range(n):
        print(i)
        accuracies = Query_by_committee(X,Y,[e1,e2,e3,e4,e5],[e11,e12,e13,e14,e15])
        al_tr = [a[0] for a in accuracies]
        al_te = [a[1] for a in accuracies]
        sv_tr = [a[2] for a in accuracies]
        sv_te = [a[3] for a in accuracies]


        trains_al.append(al_tr)
        tests_al.append(al_te)
        trains_svc.append(sv_tr)
        test_svc.append(sv_te)
    trains_al = np.array(trains_al)
    tests_al = np.array(tests_al)
    trains_svc = np.array(trains_svc)
    test_svc = np.array(test_svc)
    plt_tral = np.mean(trains_al,axis=0)
    plt_teal = np.mean(tests_al,axis=0)
    plt_trsv = np.mean(trains_svc,axis=0)
    plt_tesv = np.mean(test_svc,axis=0)

    plt_tral_var = np.var(trains_al, axis=0)
    plt_teal_var = np.var(tests_al, axis=0)
    plt_trsv_var = np.var(trains_svc, axis=0)
    plt_tesv_var = np.var(test_svc, axis=0)

    plt_tral_var = np.array([x for i,x in enumerate(plt_tral_var) if i%20==0])
    plt_teal_var = np.array([x for i,x in enumerate(plt_teal_var) if i%20==0])
    plt_trsv_var = np.array([x for i,x in enumerate(plt_trsv_var) if i%20==0])
    plt_tesv_var = np.array([x for i,x in enumerate(plt_tesv_var) if i%20==0])

    ns = range(len(plt_teal))
    plt_tral1 = np.array([x for i,x in enumerate(plt_tral) if i%20==0])
    plt_teal1 = np.array([x for i,x in enumerate(plt_teal) if i%20==0])
    plt_trsv1 = np.array([x for i,x in enumerate(plt_trsv) if i%20==0])
    plt_tesv1 = np.array([x for i,x in enumerate(plt_tesv) if i%20==0])
    ns1 = [x for i,x in enumerate(ns) if i%20==0]

    plt_tral_var_pos = plt_tral1 + plt_tral_var 
    plt_tral_var_neg = plt_tral1 - plt_tral_var
    plt_teal_var_pos = plt_teal1 + plt_teal_var
    plt_teal_var_neg = plt_teal1 - plt_teal_var
    plt_trsv_var_pos = plt_trsv1 + plt_trsv_var 
    plt_trsv_var_neg = plt_trsv1 - plt_trsv_var
    plt_tesv_var_pos = plt_tesv1 + plt_tesv_var
    plt_tesv_var_neg = plt_tesv1 - plt_tesv_var
    
    plt.ylim = (0,1)
    plt.plot(ns1,plt_tral1,color="green", marker='o',label="Active Learning Entrenamiento")
    plt.plot(ns1,plt_teal1,color="red", marker='o',label="Active Learning Prueba")
    plt.plot(ns1,plt_trsv1,color="blue", marker='o',label="SVM Entrenamiento")
    plt.plot(ns1,plt_tesv1,color="black", marker='o',label="SVM Prueba")

    plt.fill_between(ns1, plt_tral_var_pos, plt_tral_var_neg, facecolor="green", alpha=0.2)
    plt.fill_between(ns1, plt_teal_var_pos, plt_teal_var_neg, facecolor="red", alpha=0.2)
    plt.fill_between(ns1, plt_trsv_var_pos, plt_trsv_var_neg, facecolor="blue", alpha=0.2)
    plt.fill_between(ns1, plt_tesv_var_pos, plt_tesv_var_neg, facecolor="black", alpha=0.2)
    plt.legend()
    plt.title("Precisión")
    plt.show()

    trains_al = np.array([1-a for a in trains_al])
    tests_al = np.array([1-a for a in tests_al])
    trains_svc = np.array([1-a for a in trains_svc])
    test_svc = np.array([1-a for a in test_svc])
    plt_tral = np.mean(trains_al,axis=0)
    plt_teal = np.mean(tests_al,axis=0)
    plt_trsv = np.mean(trains_svc,axis=0)
    plt_tesv = np.mean(test_svc,axis=0)
    
    plt_tral_var = np.var(trains_al, axis=0)
    plt_teal_var = np.var(tests_al, axis=0)
    plt_trsv_var = np.var(trains_svc, axis=0)
    plt_tesv_var = np.var(test_svc, axis=0)

    plt_tral_var = np.array([x for i,x in enumerate(plt_tral_var) if i%20==0])
    plt_teal_var = np.array([x for i,x in enumerate(plt_teal_var) if i%20==0])
    plt_trsv_var = np.array([x for i,x in enumerate(plt_trsv_var) if i%20==0])
    plt_tesv_var = np.array([x for i,x in enumerate(plt_tesv_var) if i%20==0])
    ns = range(len(plt_teal))

    plt_tral = np.array([x for i,x in enumerate(plt_tral) if i%20==0])
    plt_teal = np.array([x for i,x in enumerate(plt_teal) if i%20==0])
    plt_trsv = np.array([x for i,x in enumerate(plt_trsv) if i%20==0])
    plt_tesv = np.array([x for i,x in enumerate(plt_tesv) if i%20==0])
    ns = [x for i,x in enumerate(ns) if i%20==0]

    plt_tral_var_pos = plt_tral + plt_tral_var 
    plt_tral_var_neg = plt_tral - plt_tral_var
    plt_teal_var_pos = plt_teal + plt_teal_var
    plt_teal_var_neg = plt_teal - plt_teal_var
    plt_trsv_var_pos = plt_trsv + plt_trsv_var 
    plt_trsv_var_neg = plt_trsv - plt_trsv_var
    plt_tesv_var_pos = plt_tesv + plt_tesv_var
    plt_tesv_var_neg = plt_tesv - plt_tesv_var

    plt.ylim = (0,1)
    plt.plot(ns,plt_tral,color="green", marker='o',label="Active Learnig Entrenamiento")
    plt.plot(ns,plt_teal,color="red", marker='o',label="Active Learnig Prueba")
    
    plt.plot(ns,plt_trsv,color="blue", marker='o',label="SVM Entrenamiento")
    plt.plot(ns,plt_tesv,color="black", marker='o',label="SVM Prueba")
    
    plt.fill_between(ns1, plt_tral_var_pos, plt_tral_var_neg, facecolor="green", alpha=0.2)
    plt.fill_between(ns1, plt_teal_var_pos, plt_teal_var_neg, facecolor="red", alpha=0.2)
    plt.fill_between(ns1, plt_trsv_var_pos, plt_trsv_var_neg, facecolor="blue", alpha=0.2)
    plt.fill_between(ns1, plt_tesv_var_pos, plt_tesv_var_neg, facecolor="black", alpha=0.2)

    plt.legend()
    plt.title("Error")
    plt.show()



X,Y = readDataset('dataset/data.json')
print(len(X))
replace_dict = {'Objetivo':2,'Negativo':0,'Positivo':3,"Neutro":2}
Y = [replace_dict[y] for y in Y]
Y = np.array(Y)
X_features = doc2vecMatrix(X)
X = X_features
#visualize_data(X,Y)
e1 = KNeighborsClassifier(n_neighbors=20)
e2 = SVC(kernel="linear",probability=True,gamma=0.001)
e3 = KNeighborsClassifier(n_neighbors=10)
e4 = SVC(kernel="rbf",probability=True,gamma=0.001)
e5 = GaussianNB()#SVC(kernel="rbf",probability=True,gamma="auto")

N_iter(X,Y)

