# 參考網站: https://wulc.me/2017/11/18/%E5%88%86%E6%89%B9%E8%AE%AD%E7%BB%83%E8%BF%87%E5%A4%A7%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import basename, join, dirname, isfile, isdir
from os import listdir, walk
import cv2
import pickle
import keras
import gc
import random


def load_data(dataset_src):
    with open(dataset_src, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()


def train_batch_generator(data_dir = './data4/'):
    data_basename = listdir(data_dir)
    Train_basename = [name for name in data_basename if (name.startswith('Train'))]
    Test_basename = [name for name in data_basename if (name.startswith('Test')) ]
    # num_items = len(Train_basename)
    print("the training order is {}".format(Train_basename))
    print("the testing order is {}".format(Test_basename))
    
    _trainX , _trainY = np.array([]), np.array([])
    _testX , _testY = np.array([]), np.array([])

    for train_file, test_file in zip(Train_basename, Test_basename):
        print('\n =====> Reading file {} and {} ...........\n'.format(data_dir + train_file, data_dir + test_file))
        # num_items -= 1
        # gc.collect()
        _trainX , _trainY = load_data(data_dir + train_file)
        _testX , _testY = load_data(data_dir + test_file)
        yield (_trainX , _trainY, _testX , _testY)
        _trainX , _trainY = np.array([]), np.array([])
        _testX , _testY = np.array([]), np.array([])
        #print('\n =====> file {} and {} finished ...........\n'.format(data_dir + train_file, data_dir + test_file))
        

def main1():
    for x_train, Y_train, x_test, Y_test in train_batch_generator(data_dir = './data/'):
        index_positive = (Y_train == 1)
        index_negative = (Y_train == 0)
        print("=> training size = {}".format(np.size(x_train,0)))
        print("=> positive size = {}".format(np.size(x_train[index_positive,:,:,:],0)))
        print("=> negative size = {}\n".format(np.size(x_train[index_negative,:,:,:],0)))
        
           
if __name__ == "__main__":
    main1()
            