import redpitaya_scpi as rpscpi
import numpy as np
import tsfresh
import time
from joblib import load

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

def absoluteEnergy(signal):
    feature = tsfresh.feature_extraction.feature_calculators.abs_energy(signal)
    
    return feature

def sumValues(signal):    
    feature = tsfresh.feature_extraction.feature_calculators.sum_values(signal)
    
    return feature

def estimate(modelName):    
    signal = pollInstanceSample()
    data = np.array(signal[refBuffer:])
    
    feature1 = absoluteEnergy(data)
    feature2 = sumValues(data)
    features = np.array([[feature1, feature2]])
    
    svmc_loaded = load(modelName)
    detected = svmc_loaded.predict(features)
    print('Class of Object Detected : ' + str(detected))
    
    return detected

def enableIndication():
    index = estimate('svm_classifier_model.joblib')    
    
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