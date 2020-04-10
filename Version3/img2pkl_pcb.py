# this program is aimed at saving image data as pkl file for PCB, connector and LCD cases
# note that PCB is gray .jpg; connector is RGB .jpg; LCD is RGB .bmp
# two-class classification considered

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import basename, join, dirname, isfile, isdir
from os import listdir, walk
import cv2
import pickle


class PCB_Img2pkl:

    def __init__(self, root, img_size):
        self.root = root
        self.CLASS_LIST = []
        self.label = []
        self.img_arr = []
        self.img_size = img_size

    def SET_CLASS_LIST(self):
        
        for dir_basename in listdir(self.root):
            dir_fullname = join(self.root,dir_basename)
            if isdir(dir_fullname):
                self.CLASS_LIST += [dir_fullname]
        #print(self.CLASS_LIST)

    def READ_IMG(self):
        init = 0
        for class_name in self.CLASS_LIST:
            print("current class: {}".format(basename(class_name)))
            for img_basename in listdir(class_name):
                img_fullname = join(class_name,img_basename)
                data = cv2.imread(img_fullname)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                if basename(class_name) == '0': self.label += [0]
                else : self.label += [1]
                self.img_arr += [data]
                init += 1
                print(init)
            print("current class: {} finished".format(basename(class_name)))
        
        self.img_arr = np.array(self.img_arr).reshape(-1,self.img_size,self.img_size,3)
        self.label = np.array(self.label)
        print("the first 2 labels = {}".format(self.label[:2]))
        print("the first 2 imgs = {}".format(self.img_arr[:2,:,:,:]))


    def SPLIT_DATA(self, test_size):
        X_train, X_test, y_train, y_test = train_test_split(self.img_arr,self.label, test_size = test_size)
        results = [(X_train,y_train),(X_test,y_test)]
        out_filename = join(self.root,'dataset_split.pkl')
        with open(out_filename,"wb") as the_file:
            pickle.dump(results, the_file)
            the_file.close()


def main1():
    root1 = r"D:\DataSet\PCB\PCB_classification"
    img_size1 = 100
    tool1 = PCB_Img2pkl(root1,img_size1)
    tool1.SET_CLASS_LIST()
    tool1.READ_IMG()
    tool1.SPLIT_DATA(0.2)
    plt.imshow(tool1.img_arr[0,:,:,:])
    plt.show()


if __name__ == "__main__":
    main1()
    


    
    

 
    