from tensorflow.keras.models import load_model
import redpitaya_scpi as rpscpi
import numpy as np
import time

rps = rpscpi.scpi('192.168.128.1')
print('Connected to Device, Acquiring Data Now...')
refBuffer = 4000
interval = 1

def pollInstanceSample():
    rps.tx_txt('ACQ:DEC 64')
    rps.tx_txt('ACQ:TRIG EXT_PE')
    rps.tx_txt('ACQ:TRIG:DLY 8192')
    rps.tx_txt('ACQ:START')
    
    while 1:
        rps.tx_txt('ACQ:TRIG:STAT?')
        if rps.rx_txt() == 'TD':
            break
        
    rps.tx_txt('ACQ:SOUR1:DATA?')    
    strData = rps.rx_txt()[1:-1]  
    arrData = np.fromstring(strData,dtype=float,sep=',')
    
    return arrData

def estimate(modelName):    
    x = []
    
    for i in range(5):
        signal = pollInstanceSample()
        data = signal[refBuffer:]
        x.append(data)
        
    x_pred = np.vstack(x)
    x_pred = np.reshape(x_pred, (x_pred.shape[0], 1, x_pred.shape[1]))
    model = load_model(modelName)
    yhat = model.predict(x_pred)
    print(yhat)
    indices = []
    
    for i in range(yhat.shape[0]):
        maxIndex = np.where(yhat[i] == np.amax(yhat[i]))[0]
        indices.append(maxIndex)

    t = np.vstack(indices)
    class1 = (np.count_nonzero(t == 0))
    class2 = (np.count_nonzero(t == 1))
    class3 = (np.count_nonzero(t == 2))

    prob = np.array([class1,class2,class3])
    detected = np.where(prob == np.amax(prob))[0][0]
    print('Class of Object Detected : ' + str(detected))
    
    return detected

def enableIndication():
    index = estimate('lstm_model.h5')
    
    if index == 0:
        rps.tx_txt('DIG:PIN LED1' + ',' + str(1))  
        rps.tx_txt('DIG:PIN LED2' + ',' + str(0))  
        rps.tx_txt('DIG:PIN LED3' + ',' + str(0))   
    elif index == 1:
        rps.tx_txt('DIG:PIN LED1' + ',' + str(0))  
        rps.tx_txt('DIG:PIN LED2' + ',' + str(1))  
        rps.tx_txt('DIG:PIN LED3' + ',' + str(0))
    elif index == 2:
        rps.tx_txt('DIG:PIN LED1' + ',' + str(0))  
        rps.tx_txt('DIG:PIN LED2' + ',' + str(0))  
        rps.tx_txt('DIG:PIN LED3' + ',' + str(1))

if __name__=='__main__':
    while 1:
        enableIndication()
        time.sleep(interval)