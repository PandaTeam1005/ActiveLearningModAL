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
from utils import readDataset
from vectorize import *
from copy import deepcopy
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
    while len(X_pool)>1:
        query_index, _ = learner.query(X_pool)
        x, y = X_pool[query_index].reshape(1, -1), Y_pool[query_index].reshape(1, )
        learner.teach(X=x, y=y)
        X_pool, Y_pool = np.delete(X_pool, query_index, axis=0), np.delete(Y_pool, query_index)
        model_accuracy =1 - learner.score(X_pool, Y_pool)
        
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
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
    
    X_pool = deepcopy(np.delete(X_train,indexs,axis=0))
    Y_pool = deepcopy(np.delete(Y_train,indexs))
    for i in range(len(estimator)):
        learners.append(ActiveLearner(estimator=estimator[i],X_training=X_0,y_training=Y_0))
    committee = Committee(learner_list=learners)
    
    
    index = 0
    plts_train = []
    plts_test = []
    while len(X_pool)>=1:
        query_indxs,_ = committee.query(X_pool)
        committee.teach(X=X_pool[query_indxs],y=Y_pool[query_indxs])
        X_0 = np.append(X_0,X_pool[query_indxs],axis=0)
        ver = Y_pool[query_indxs]
        Y_0 = np.append(Y_0,Y_pool[query_indxs][0])
        X_pool = np.delete(X_pool,query_indxs,axis=0)
        Y_pool = np.delete(Y_pool,query_indxs,axis=0)
        
        model_accuracy = 1- committee.score(X_0,Y_0)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        index+=1
        predicts = committee.predict(X_test)
        corrects = (predicts==Y_test)
        accs =1 - sum([1 if i else 0 for i in corrects])/len(predicts)
        print(accs)
        plts_train.append(model_accuracy)
        plts_test.append(accs)
    return (committee,plts_test,plts_train)




def Query_by_committee(X,Y,estimator):
    indexs = np.random.randint(low=0, high=len(X), size=int(len(X)/5))
    X_test = X[indexs]
    Y_test = Y[indexs]
    X_train = np.delete(X,indexs,axis=0)
    Y_train = np.delete(Y,indexs)
    indexs = np.random.randint(0,len(X_train),50)
    X_0 = X[indexs]
    Y_0 = Y[indexs]
    learner,test,train = cmte_loop(estimator,X_0,Y_0,X_train,Y_train,X_test,Y_test,indexs)
    predictions = learner.predict(X_test)
    is_Correct = (predictions==Y_test)
    accs = sum([1 if i else 0 for i in is_Correct])/len(predictions)
    print(accs)
    ns = range(len(train))
    plt.ylim = (0,1)
    plt.plot(ns,train,color="green")
    plt.plot(ns,test,color="red")
    plt.show()






#iris = load_iris()

#X = iris['data']


X, Y = readDataset('dataset/data.json')
replace_dict = {'Objetivo':2,'Negativo':0,'Positivo':3,"Neutro":0}
Y = [replace_dict[y] for y in Y]
Y = np.array(Y)
X_features = doc2vecMatrix(X)
X = X_features
visualize_data(X,Y)
e1 = KNeighborsClassifier(n_neighbors=20)
e2 = SVC(kernel="linear",probability=True,gamma="auto")
e3 = KNeighborsClassifier(n_neighbors=10)
e4 = SVC(kernel="rbf",probability=True,gamma="auto")
e5 = GaussianNB()#SVC(kernel="rbf",probability=True,gamma="auto")
Query_by_committee(X,Y,[e1,e2,e3,e4,e5])

