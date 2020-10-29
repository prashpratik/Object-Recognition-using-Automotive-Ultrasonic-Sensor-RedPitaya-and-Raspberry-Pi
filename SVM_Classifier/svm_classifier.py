import numpy as np
from numpy import genfromtxt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump, load

objClass = ['Wall', 'Human', 'Car']

def genDataSet():
    y = []
    x = []
    
    for i in range(3):
        signal = np.array(genfromtxt('Data/'+ objClass[i] +'Features.csv',delimiter=';'))
        x.append(signal)
        
        for j in range(signal.shape[0]):
            y.append(i)
            
    x = np.vstack(x)
    y = np.array(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    return x_train, x_test, y_train, y_test

def svmcModelGen(x_train,y_train):    
    svmc = svm.SVC(kernel='linear')
    svmc.fit(x_train, y_train)
    
    dump(svmc,'svm_classifier_model.joblib') 
    
def estimate(modelName,inputSignal):  
    svmc_loaded = load(modelName) 
    y_pred = svmc_loaded.predict(inputSignal)
    
    return y_pred

if __name__=='__main__':
    x_train, x_test, y_train, y_test = genDataSet()
    
    print('Shape of x_train:'+ str(x_train.shape))
    print('Shape of y_train:'+ str(y_train.shape))
    print('Shape of x_test:'+ str(x_test.shape))
    print('Shape of y_test:'+ str(y_test.shape))
    
    svmcModelGen(x_train,y_train)
    
    y_pred = estimate('svm_classifier_model.joblib',x_test)
    
    print('Accuracy:',metrics.accuracy_score(y_test, y_pred))