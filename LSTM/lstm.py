import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,Dropout,LSTM
import numpy as np
from numpy import genfromtxt

objClass = ['Wall', 'Human', 'Car']
refBuffer = 4000

def genTrainSet():
    y = []
    x = []
    
    for i in range(3):
        signal = np.array(genfromtxt('Data/Train/'+ objClass[i] +'.csv',delimiter=';'))
        data = np.delete(signal,np.s_[:refBuffer],1)
        x.append(data)
        
        for j in range(signal.shape[0]):
            y.append(i)
            
    x_train = np.vstack(x)
    y_train = np.vstack(y)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])) 
    
    return x_train,y_train

def genValSet():
    y = []
    x = []
    
    for i in range(3):
        signal = np.array(genfromtxt('Data/Validation/'+ objClass[i] +'.csv',delimiter=';'))
        data = np.delete(signal,np.s_[:refBuffer],1)
        x.append(data)
        
        for j in range(signal.shape[0]):
            y.append(i)
            
    x_val = np.vstack(x)
    y_val = np.vstack(y)
    x_val = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1])) 
    
    return x_val,y_val

def genTestSet():
    y = []
    x = []
    
    for i in range(3):
        signal = np.array(genfromtxt('Data/Test/'+ objClass[i] +'.csv',delimiter=';'))
        data = np.delete(signal,np.s_[:refBuffer],1)
        x.append(data)
        
        for j in range(signal.shape[0]):
            y.append(i)
            
    x_test = np.vstack(x)
    y_test = np.vstack(y)
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    
    return x_test,y_test

def lstmModelGen(x_train,y_train,x_val,y_val):
    model = Sequential()
    
    model.add(LSTM(128,input_shape=(x_train.shape[1:]),activation='relu',return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(3,activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)

    model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    model.fit(x_train,y_train,epochs=10,validation_data=(x_val,y_val))

    model.save('lstm_model.h5')

def estimate(modelName,inputSignal):
    model = load_model(modelName)
    yhat = model.predict(inputSignal)
    
    indices = []
    for i in range(yhat.shape[0]):
        maxIndex = np.where(yhat[i] == np.amax(yhat[i]))[0]
        indices.append(maxIndex)
        
        print('\nClass of Object Detected for Signal ' + str(i+1) + ' :')
        print('Wall = ' + str(round(yhat[i][0]*100,2)) + '%')
        print('Human = ' + str(round(yhat[i][1]*100,2)) + '%')
        print('Car = ' + str(round(yhat[i][2]*100,2)) + '%')

    y_pred = np.vstack(indices)
    
    return y_pred


if __name__=='__main__':
    x_train,y_train = genTrainSet()
    x_val,y_val = genValSet()
    x_test,y_test = genTestSet()
    
    print('Shape of x_train:'+ str(x_train.shape))
    print('Shape of y_train:'+ str(y_train.shape))
    print('Shape of x_val:'+ str(x_val.shape))
    print('Shape of y_val:'+ str(y_val.shape))
    print('Shape of x_test:'+ str(x_test.shape))
    print('Shape of y_test:'+ str(y_test.shape))
    print('\n')
    
    lstmModelGen(x_train,y_train,x_val,y_val)
    
    y_pred = estimate('lstm_model.h5',x_test)
    
    test_acc = round(len(np.where(np.transpose(y_test)[0]==np.transpose(y_pred)[0])[0])/len(np.transpose(y_test)[0]),4)

    print('\nTest accuracy:', test_acc)