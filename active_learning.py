from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.svm import SVC
from modAL.disagreement import max_disagreement_sampling
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
def visualize_data(X,Y):
    pca = PCA(n_components=2)
    t_X = pca.fit_transform(X=X)
    x_toplt,y_toplt = t_X[:,0],t_X[:,1]
    #plt.figure(figsize=(8.5, 6), dpi=130)
    #plt.scatter(x=x_toplt, y=y_toplt, c=Y, cmap='viridis', s=50, alpha=8/10)
    #plt.title('Text Classification after PCA transformation')
    #plt.show()
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
    
    learner.predict(X_pool)
    while index<30:
        query_index, _ = learner.query(X_pool)
        x, y = X_pool[query_index].reshape(1, -1), Y_pool[query_index].reshape(1, )
        learner.teach(X=x, y=y)
        X_pool, Y_pool = np.delete(X_pool, query_index, axis=0), np.delete(Y_pool, query_index)
        model_accuracy = learner.score(X_pool, Y_pool)
        
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        
        predicts = learner.predict(X_test)
        corrects = (predicts==Y_test)
        accs = sum([1 if i else 0 for i in corrects])/len(predicts)
        print(accs)
        index+=1
    return learner

def activeLearner(X,Y,estimator):
    X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.33)
    indexs = np.random.randint(0,len(X_train),3)
    X_0 = X[indexs]
    Y_0 = Y[indexs]
    learner = al_Loop(estimator,X_0,Y_0,X_train,Y_train,X_test,Y_test,indexs)
    predictions = learner.predict(X_test)
    is_Correct = (predictions==Y_test)
    






iris = load_iris()

X = iris['data']
Y = iris['target']

activeLearner(X,Y,SVC(kernel="rbf",probability=True,gamma="auto"))