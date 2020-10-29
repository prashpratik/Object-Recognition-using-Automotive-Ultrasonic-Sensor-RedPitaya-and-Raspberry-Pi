import numpy as np
from numpy import genfromtxt
from sklearn.naive_bayes import GaussianNB
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

def nbcModelGen(x_train,y_train):    
    nbc = GaussianNB()
    nbc.fit(x_train, y_train)
    
    dump(nbc,'bayes_classifier_model.joblib') 
    
def estimate(modelName,inputSignal):  
    nbc_loaded = load(modelName) 
    y_pred = nbc_loaded.predict(inputSignal)
    
    return y_pred   

if __name__=='__main__':
    x_train, x_test, y_train, y_test = genDataSet()
    
    print('Shape of x_train:'+ str(x_train.shape))
    print('Shape of y_train:'+ str(y_train.shape))
    print('Shape of x_test:'+ str(x_test.shape))
    print('Shape of y_test:'+ str(y_test.shape))
    
    nbcModelGen(x_train,y_train)
    
    y_pred = estimate('bayes_classifier_model.joblib',x_test)
    
    print('Accuracy:',metrics.accuracy_score(y_test, y_pred))