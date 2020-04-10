# The program is only treating training data
# we extract data from training.pickle by fixing the numbers of testing data

import pandas as  pd
import numpy as np
from os.path import basename, join, dirname, isfile, isdir
from os import listdir, walk
import shutil
import random
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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

#4/6 updated
def extract_data(X,Y,num_data = 10):
    ind_tot = range(np.size(X,0))
    ind_sub = random.sample(ind_tot,num_data)
    return X[ind_sub,:,:,:], Y[ind_sub]

#4/8 updated
def extract_data_pn(X,Y,p_num_data = 10, n_num_data = 10):
    X_positive, Y_positive, X_negative, Y_negative = X[Y==1,:,:,:], Y[Y==1] , X[Y==0,:,:,:], Y[Y==0]
    X_positive, Y_positive, X_negative, Y_negative = X_positive[:p_num_data,:,:,:], Y_positive[:p_num_data], X_negative[:n_num_data,:,:,:], Y_negative[:n_num_data]
    X = np.concatenate((X_positive,X_negative), axis = 0)
    Y = np.concatenate((Y_positive,Y_negative), axis = 0)
    X, Y = shuffle(X,Y, random_state = 0)
    return X, Y

#4/9 updated
def merge_pkl():
    pass


#4/8 updated  
def main1():
    src = "D:\\DataSet\\PCB\\data\\" #extracted from src
    dst = "./data5/" #saved to dst

    num_positive_samples = 0
    num_negative_samples = 0

    for pkl_basename in listdir(src):

        src_filename = src + pkl_basename
        dst_filename = dst + pkl_basename
        
        if pkl_basename.startswith("Train-"):
            _trainX , _trainY = load_data(src_filename)
            #_trainX, _trainY = extract_data(_trainX, _trainY, 314)
            _trainX, _trainY = extract_data_pn(_trainX, _trainY, 5029, 2514)
            num_positive_samples += np.sum(_trainY==1)
            num_negative_samples += np.sum(_trainY==0)

            _train = [_trainX,_trainY]
            save_data(dst_filename, _train)
            print("%s finished\n" % pkl_basename)
        else:
            shutil.copyfile(src_filename,dst_filename)

    print("the total positive samples = %d " % num_positive_samples)
    print("the total negative samples = %d " % num_negative_samples)

#4/9 updated
def main2():
    src = "D:\\DataSet\\PCB\\data4\\" #extracted from src
    dst = "./data6/" #saved to dst
    
    count_train = 0
    count_test = 0
    num_positive_samples = 0
    num_negative_samples = 0
    
    num_test_samples = 0
    
    tot_trainX , tot_trainY = np.array([]), np.array([])
    tot_testX , tot_testY = np.array([]), np.array([])
    
    for pkl_basename in listdir(src):

        src_filename = src + pkl_basename
        dst_filename = dst + "dataset_split.pkl"
        
        if pkl_basename.startswith("Train-"):
            _trainX , _trainY = load_data(src_filename)
            num_positive_samples += np.sum(_trainY==1)
            num_negative_samples += np.sum(_trainY==0)
            
            if count_train == 0 : tot_trainX, tot_trainY = _trainX , _trainY 
            else: tot_trainX, tot_trainY = np.concatenate((tot_trainX, _trainX),axis = 0), np.concatenate((tot_trainY, _trainY),axis = 0)

            count_train += 1
            print("%s finished\n" % pkl_basename)
            
        else:
            _testX , _testY = load_data(src_filename)
            num_test_samples += len(_testY)
            
            if count_test == 0 : tot_testX, tot_testY = _testX , _testY
            else: tot_testX, tot_testY = np.concatenate((tot_testX, _testX),axis = 0), np.concatenate((tot_testY, _testY),axis = 0)
            
            count_test += 1
            
        data = [(tot_trainX , tot_trainY), (tot_testX, tot_testY)]
        save_data(dst_filename, data)

    print("the total positive samples = %d " % num_positive_samples)
    print("the total negative samples = %d " % num_negative_samples)
    print("the total test samples = %d " % num_test_samples)
    
# 4/10 update (pick data from a pkl file)
def main3():
    
    def prt_config(_trainY):
        num_positive_samples = np.sum(_trainY==1)
        num_negative_samples = np.sum(_trainY==0)
        print("total num of data = %d " % (num_positive_samples + num_negative_samples))
        print("the total positive samples = %d " % num_positive_samples)
        print("the total negative samples = %d " % num_negative_samples)
        print("==> pn-ratio = %f" % (num_positive_samples/num_negative_samples))
    
    src_filenames = ["D:\\Datasets\\Connector_classification\\dataset_split.pkl"]*3
    dst_filenames = [".\\d0.7k_pn1.9\\dataset_split.pkl",
                    ".\\d1.2k_pn1.9\\dataset_split.pkl",
                    ".\\d175_pn1.9\\dataset_split.pkl"]
    num_data_list = [700,1224,175]
    
    
    for src_filename, dst_filename, num_data in zip(src_filenames,dst_filenames, num_data_list):
    
        (_trainX , _trainY), (_testX, _testY) = load_data(src_filename)
        prt_config(_trainY) #before
        
        _trainX, _trainY = extract_data(_trainX, _trainY, num_data)
        prt_config(_trainY) #after
        
        data = [(_trainX , _trainY), (_testX, _testY)] 
        save_data(dst_filename, data)
        print("finished................\n")
    
    

if __name__ == "__main__":
    #main1()
    #main2()
    main3()
