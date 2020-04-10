import pandas as  pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle
import math

def load_data(dataset_src):
    with open(dataset_src, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()

def save_data(dataset_dst,data):
    with open(dataset_dst,"wb") as the_file:
        pickle.dump(data,the_file)
        the_file.close()


path = r".\dataset_split_pcb.pkl"

(x_train,y_train),(x_test,y_test) = load_data(path)
n_cut = 16
L_train = x_train.shape[0]
L_test = x_test.shape[0]

print("the total training data = {} ".format(L_train))
print("the total testing data = {} ".format(L_test))

interval_train = math.ceil(L_train/n_cut)
interval_test = math.ceil(L_test/n_cut)

for i in range(n_cut):
    _trainX = x_train[i*interval_train:(i+1)*interval_train,:,:,:]
    _trainY = y_train[i*interval_train:(i+1)*interval_train]
    
    _testX = x_test[i*interval_test:(i+1)*interval_test,:,:,:]
    _testY = y_test[i*interval_test:(i+1)*interval_test]
    
    _train = [_trainX,_trainY]
    _test = [_testX,_testY]

    outfile_name_train = 'Train-' + str(i) + '.pkl'
    outfile_name_test = 'Test-' + str(i) + '.pkl'

    save_data(outfile_name_train, _train)
    save_data(outfile_name_test, _test)
    print("data %d finished" % i)
