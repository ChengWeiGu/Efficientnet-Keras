import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import basename, join, dirname, isfile, isdir
from os import listdir, walk
import cv2
import pickle
import ImagPreprocess as imp

class AutoData:

    def __init__(self, path):
        
        self.src = path
        self.filenames = []
        self.labels = []
        self.filenames_dst = join(self.src,"filenames.csv")
        self.dataset_dst = join(self.src,"datasets.pkl")
        self.dataset_split_dst = join(self.src,"datasets_split.pkl")
        self.imag_size = 160

        if (not isfile(self.filenames_dst)):
            label = 0
            for d in listdir(self.src):
                path_t = join(self.src,d)
                if isdir(path_t):
                    for f in listdir(path_t):
                        self.filenames += [join(path_t,f)]
                        self.labels += [label]
                label = label + 1

            #將路徑與數據標記儲存CSV
            with open(self.filenames_dst, 'w') as the_file:
                for fn, l in zip(self.filenames, self.labels):
                    the_file.write(fn + "," + str(l) + "\n")
                the_file.close()    

    def im2pickle(self, test_size = 0.33):

        if (isfile(self.dataset_dst) and isfile(self.dataset_split_dst)): print("the dataset and dataset_split have existed!") 
        if (not (isfile(self.dataset_dst) and isfile(self.dataset_split_dst))):

            df_file = pd.read_csv(self.filenames_dst, header = None ,encoding = 'utf-8')
            df_file.columns = ['filenames','label']
            print(df_file)
            data = []        
            for p in df_file['filenames'].values:
                imag = cv2.imread(p)
                imag = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
                data += [list(imag.ravel())]
            #壓縮全部數據並儲存
            data_all = (np.array(data), df_file['label'].values) 
            with open(self.dataset_dst,"wb") as the_file:
                pickle.dump(data_all,the_file)
                the_file.close()
            #壓縮訓練與測試數據並儲存
            X_train, X_test, y_train, y_test = train_test_split(data_all[0],data_all[1], test_size = test_size, random_state = 87)
            data_all = [(X_train,y_train),(X_test,y_test)]
            with open(self.dataset_split_dst,"wb") as the_file:
                pickle.dump(data_all,the_file)
                the_file.close()


    def load_data(self):
            with open(self.dataset_dst, "rb") as the_file:
                return pickle.load(the_file)
                the_file.close()


    def load_split_data(self):
            with open(self.dataset_split_dst, "rb") as the_file:
                return pickle.load(the_file)
                the_file.close()



if __name__ == "__main__":

    src = r".\dataset\"
    AD = AutoData(src)
    AD.im2pickle(test_size = 0.2)
    
    X, y = AD.load_data()
    print("the total data shape = ",X.shape)
    print("the label = ",y)

    (X_train,y_train),(X_test,y_test) = AD.load_split_data()
    print("training data shape = ",X_train.shape)
    print("training label = ",y_train)
    print("testing data shape = ",X_test.shape)
    print("testing label = ",y_test)
    
    plt.imshow(X_train[0,:].reshape(-1,AD.imag_size), cmap = plt.cm.gray)
    plt.show()
    

 
    
