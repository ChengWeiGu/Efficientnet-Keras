import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from keras.datasets import mnist
from efficientnet.keras import EfficientNetB0
from efficientnet.keras import EfficientNetB3
from efficientnet.keras import EfficientNetB5
from efficientnet.keras import EfficientNetB7
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , Conv2D, MaxPooling2D
from keras.models import Model
from keras import optimizers, losses
from keras.utils import np_utils
from keras.utils import plot_model
from keras import backend as K
from keras import metrics
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
# from keras.utils import multi_gpu_model
import cv2
import os
import json
import argparse
import pickle
from sklearn.metrics import confusion_matrix
from DataGen import train_batch_generator


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, mse ,acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {:.4f}, acc: {:.4f}\n'.format(loss, acc))


#計算recall and precision by confusing matrix
def recall_precision(matrix, num_cls = 2):
    s = np.size(matrix,0)
    recall = []
    precision = []
    acc_tot = np.diagonal(matrix).sum()/np.sum(matrix)

    for i in range(s):

        if (np.sum(matrix[:,i]) == 0): precision += [0.0]
        else: precision += [matrix[i,i]/np.sum(matrix[:,i])]
        
        if (np.sum(matrix[i,:]) == 0): recall += [0.0]
        else: recall += [matrix[i,i]/np.sum(matrix[i,:])]

    if num_cls > 2:
        res = [acc_tot] + precision + recall
    elif num_cls == 2:
        res = [acc_tot] + [precision[1], recall[1]]

    return np.array(res)
    


def load_data(dataset_dst):
    with open(dataset_dst, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()


class model_selection:
    
    def __init__(self, model_size , imag_size, num_cls):
    
        try: 
            if model_size == "B0": model = EfficientNetB0(weights = 'imagenet', input_shape = (imag_size,imag_size,3), include_top = False)
            elif model_size == "B3": model = EfficientNetB3(weights = 'imagenet', input_shape = (imag_size,imag_size,3), include_top = False)
            elif model_size == "B5": model = EfficientNetB5(weights = 'imagenet', input_shape = (imag_size,imag_size,3), include_top = False)
            elif model_size == "B7": model = EfficientNetB7(weights = 'imagenet', input_shape = (imag_size,imag_size,3), include_top = False)

            ENet_out = model.output
            ENet_out = Flatten()(ENet_out)


            Hidden1_in = Dense(1024, activation="relu")(ENet_out)
            Hidden1_in = Dropout(0.5)(Hidden1_in)

            predictions = Dense(units = num_cls, activation="softmax")(Hidden1_in)
            self.model_f = Model(input = model.input, output = predictions)
            self.model_f.compile(optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy])
        
        except:
            print("only B0/B3/B5/B7 allowed")
            
    def get_model(self):
        return self.model_f




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "To start training model"
    parser.add_argument("-t", "--train", help=help_, action='store_true')
    help_ = "To select the model size of efficientnet"
    parser.add_argument("-m", "--model_size", help= help_)
    help_ = "To set the image_size"
    parser.add_argument("-i", "--imag_size", help= help_)
    help_ = "To set the num_cls"
    parser.add_argument("-c", "--num_cls", help= help_)
    
    args = parser.parse_args()
    
    model_size = args.model_size
    imag_size = int(args.imag_size)
    num_cls = int(args.num_cls)
    
    select_obj = model_selection(model_size = model_size, imag_size = imag_size, num_cls = num_cls)
    model_f = select_obj.get_model()
    # model_f.summary()
    # plot_model(model_f, to_file='ENetB0.png', show_shapes=True)
    
    #----------------fitting params control------------------#
    epochs = 50
    batch_size = 24
    validation_split=0.1
    shuffle=True
    verbose=1
    count = 0

    w_filename = 'efficientnetB5_last.h5' #跑完最後存檔
    history_name = 'historyB5' #存每一步的acc
    #----------------fitting params control------------------#

    # Train model
    if args.train: 
        #每一次存最好的一次權重: monitor = loss or val_loss or acc or val_acc
        checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', verbose=verbose,
                                  save_best_only=True, mode='auto', period=1)

        
        for x_train, Y_train, x_test, Y_test in train_batch_generator():
            
            print("x_train.shape = {}".format(x_train.shape))
            print("For Train, PASS: {}; NG: {}\n".format(x_train.shape[0]-np.sum(Y_train),np.sum(Y_train)))
            print("x_test.shape = {}".format(x_test.shape))
            print("For Test, PASS: {}; NG: {}\n".format(x_test.shape[0]-np.sum(Y_test),np.sum(Y_test)))

            x_train = x_train.astype('float32') / 255
            x_test = x_test.astype('float32') / 255
            Y_train = np_utils.to_categorical(Y_train, num_cls)
            Y_test = np_utils.to_categorical(Y_test, num_cls)
            
            history = model_f.fit(x_train, Y_train,
                    epochs=epochs,
                    batch_size = batch_size,
                    validation_data=(x_test,Y_test),
                    shuffle=shuffle,
                    verbose=verbose,
                    callbacks=[checkpoint, TestCallback((x_test, Y_test))])
            
            history_filename = history_name + "-{}.json".format(count)
            with open(history_filename, 'w') as f:
                json.dump(history.history, f)
                f.close()
            
            count += 1

        
        model_f.save_weights(w_filename)
        
        
    # load weights and do the prediction
    if args.weights:
        model_f.load_weights(args.weights)
        train_matrix = np.matrix(np.zeros((2,2)))
        test_matrix = np.matrix(np.zeros((2,2)))

        for x_train, y_train, x_test, y_test in train_batch_generator():
            x_train = x_train.astype('float32') / 255
            x_test = x_test.astype('float32') / 255
            
            #test
            test_predictions = model_f.predict(x_test)
            y_test_pre = np.argmax(test_predictions,axis = 1)
            test_matrix = test_matrix + confusion_matrix(y_test, y_test_pre)
            
            #train
            train_predictions = model_f.predict(x_train)
            y_train_pre = np.argmax(train_predictions,axis = 1)
            train_matrix = train_matrix + confusion_matrix(y_train, y_train_pre)

        print("test matrix = ", test_matrix)
        print("train matrix = ", train_matrix)
        for md, name in zip([test_matrix,train_matrix],["test_matrix.csv","train_matrix.csv"]):
            df_md = pd.DataFrame(md)
            df_md.to_csv(name)


        #Save recall and precision
        a = recall_precision(train_matrix, num_cls = 2)
        b = recall_precision(test_matrix, num_cls = 2)
        c = np.vstack((a,b))
        print(c)
        pd.DataFrame(c).to_csv("results_by_matrix.csv")
