from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from modAL.models import ActiveLearner,Committee
from modAL.uncertainty import uncertainty_sampling
from sklearn.svm import SVC
from modAL.disagreement import max_disagreement_sampling
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import numpy as np
import random
from utils import readDataset
from vectorize import *
from copy import deepcopy
from sklearn.metrics import accuracy_score
def visualize_data(X,Y):
    
    pca = PCA(n_components=2)
    t_X = pca.fit_transform(X=X)
    x_toplt,y_toplt = t_X[:,0],t_X[:,1]
    plt.figure(figsize=(8.5, 6), dpi=130)
    plt.scatter(x=x_toplt, y=y_toplt, c=Y, cmap='viridis', s=50, alpha=8/10)
    plt.title('Text Classification after PCA transformation')
    plt.show()
    return (x_toplt,y_toplt)

# def select_training_set(X,Y):
#     N = X.shape[0]
#     #train_indices = 


#asumir q X_train esta al principio de X
#seleccionar bien los examples
def al_Loop(estimator,X_train,Y_train,X,Y,X_test,Y_test,indexs):
    learner = ActiveLearner(estimator=estimator, X_training=X_train, y_training=Y_train)
    X_pool = np.delete(X,indexs, axis=0)
    Y_pool = np.delete(Y,indexs, axis=0)
    index = 0

    accuracy  = 0
    while len(X_pool)>0:
        query_index, _ = learner.query(X_pool)
        x, y = X_pool[query_index].reshape(1, -1), Y_pool[query_index].reshape(1, )
        learner.teach(X=x, y=y)
        X_pool, Y_pool = np.delete(X_pool, query_index, axis=0), np.delete(Y_pool, query_index)
        model_accuracy =1 - learner.score(X_pool, Y_pool)
        
        print('Error after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        accuracy = model_accuracy
        predicts = learner.predict(X_test)
        corrects = (predicts==Y_test)
        accs = (sum([1 if i else 0 for i in corrects])/len(predicts))
        accs = 1 - accs
        print(accs)
        index+=1
    return learner

def activeLearner(X,Y,estimator):
    #X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.33)
    indexs = range(265)
    X_test = X[indexs]
    Y_test = Y[indexs]
    X_train = np.delete(X,indexs,axis=0)
    Y_train = np.delete(Y,indexs)
    indexs = np.random.randint(0,len(X_train),10)
    X_0 = X[indexs]

    Y_0 = Y[indexs]
    learner = al_Loop(estimator,X_0,Y_0,X_train,Y_train,X_test,Y_test,indexs)
    predictions = learner.predict(X_test)
    is_Correct = (predictions==Y_test)
    accs = sum([1 if i else 0 for i in is_Correct])/len(predictions)
    print(accs)
    
def cmte_loop(estimator,X_0,Y_0,X_train,Y_train,X_test,Y_test,indexs,n=5):
    learners = []
    print(len(indexs))
    X_pool = deepcopy(np.delete(X_train,indexs,axis=0))
    Y_pool = deepcopy(np.delete(Y_train,indexs))
    for i in range(len(estimator)):
        learners.append(ActiveLearner(estimator=estimator[i],X_training=X_0,y_training=Y_0))
    committee = Committee(learner_list=learners)
    index = 0
    accuracies = []
    while len(X_pool)>0:
        query_indxs,_ = committee.query(X_pool)
        if len(query_indxs)>1:
            raise Exception("NOOOOOOOOOOOOOOO")
        committee.teach(X=X_pool[query_indxs],y=Y_pool[query_indxs])
        X_0 = np.append(X_0,X_pool[query_indxs],axis=0)
        Y_0 = np.append(Y_0,Y_pool[query_indxs][0])
        X_pool = np.delete(X_pool,query_indxs,axis=0)
        Y_pool = np.delete(Y_pool,query_indxs,axis=0)
        accuracies.append(evaluate(committee,X_0,Y_0,X_test,Y_test))
        #model_accuracy = 1- committee.score(X_0,Y_0)
        #print('Error after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        index+=1
        #predicts = committee.predict(X_test)
        #corrects = (predicts==Y_test)
        #accs =1 - sum([1 if i else 0 for i in corrects])/len(predicts)
        #print(accs)
        #plts_train.append(model_accuracy)
        #plts_test.append(accs)

    return (committee,accuracies)
def evaluate(commitee,X,Y,X_test,Y_test):
    e = SVC(gamma=0.001)
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



def Query_by_committee(X,Y,estimator):
    X = deepcopy(X)
    Y = deepcopy(Y)
    print(len(X))
    
    indexs = np.random.choice(range(len(X)),size=60,replace=False)
    X_test = X[indexs]
    Y_test = Y[indexs]
    X_train = np.delete(X,indexs,axis=0)
    Y_train = np.delete(Y,indexs)
    print(len(X_train))
    indexs = np.random.choice(range(len(X_train)),size=50,replace=False)
    X_0 = X[indexs]
    Y_0 = Y[indexs]
    _,accuracies = cmte_loop(estimator,X_0,Y_0,X_train,Y_train,X_test,Y_test,indexs)
    #predictions = learner.predict(X_test)
    #accs = accuracy_score(Y_test,predictions)
    #print(accs)    
    #ns = range(len(train))
    #plt.ylim = (0,1)
    #plt.plot(ns,train,color="green")
    #plt.plot(ns,test,color="red")
    #plt.show()
    return accuracies

def estimator_loop(X_test,Y_test,X_train,Y_train,estimator):
    X_pool = deepcopy(X_train)
    Y_pool = deepcopy(Y_train)
    
    indexs = np.random.randint(0,len(X_train),50)
    X_0 = X_train[indexs]
    Y_0 = Y_train[indexs]
    train = []
    test = []
    
    while len(X_pool>0):
        print(len(X_pool))
        X_pool = np.delete(X_pool,indexs,axis=0)
        Y_pool = np.delete(Y_pool,indexs)
        estimator = SVC(kernel="rbf",gamma=0.001)
        estimator.fit(X_0,Y_0)
        predicts = estimator.predict(X_0)
        try:
            indexs = np.random.randint(0,len(X_pool),1)
        except:
            indexs = []
        train.append(1-accuracy_score(Y_0,predicts))
        predicts = estimator.predict(X_test)
        test.append(1-accuracy_score(Y_test,predicts))
        
        X_0 = np.append(X_0,X_pool[indexs],axis=0)
        Y_0 = np.append(Y_0,Y_pool[indexs])
    return (train,test)
        
        
      
def N_iter(X,Y,estimator,n=30):
    trains_al = []
    tests_al = []
    trains_svc = []
    test_svc = []

    for i in range(n):
        accuracies = Query_by_committee(X,Y,estimator)
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
    ns = range(len(plt_teal))
    plt.ylim = (0,1)
    plt.plot(ns,plt_tral,color="green")
    plt.plot(ns,plt_teal,color="red")
    
    plt.plot(ns,plt_trsv,color="blue")
    plt.plot(ns,plt_tesv,color="black")
    plt.show()

    



#iris = load_iris()

#X = iris['data']


X,Y = readDataset('dataset/data.json')
replace_dict = {'Objetivo':2,'Negativo':0,'Positivo':3,"Neutro":1}
Y = [replace_dict[y] for y in Y]
Y = np.array(Y)
X_features = doc2vecMatrix(X)
X = X_features
visualize_data(X,Y)
e1 = KNeighborsClassifier(n_neighbors=20)
e2 = SVC(kernel="linear",probability=True,gamma=0.001)
e3 = KNeighborsClassifier(n_neighbors=10)
e4 = SVC(kernel="rbf",probability=True,gamma=0.001)
e5 = GaussianNB()#SVC(kernel="rbf",probability=True,gamma="auto")

N_iter(X,Y,[e1,e2,e3,e4,e5])

