import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from keras.datasets import mnist
# from efficientnet import EfficientNetB0
# from efficientnet import EfficientNetB3
from efficientnet import EfficientNetB5
# from efficientnet import EfficientNetB7
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , Conv2D, MaxPooling2D
from keras.models import Model
from keras import optimizers, losses
from keras.utils import np_utils
from keras.utils import plot_model
from keras import backend as K
from keras import metrics
from keras.callbacks import ModelCheckpoint
# from keras.utils import multi_gpu_model
import cv2
import os
import json
import argparse
import pickle
from sklearn.metrics import confusion_matrix

#計算recall and precision by confusing matrix
def recall_precision(matrix):
    s = np.size(matrix,0)
    recall = []
    precision = []
    acc_tot = np.diagonal(matrix).sum()/np.sum(matrix)
    for i in range(s):
        if (np.sum(matrix[:,i]) == 0): precision += [100]
        else: precision += [matrix[i,i]/np.sum(matrix[:,i])]
        
        if (np.sum(matrix[:,i]) == 0): recall += [100]
        else: recall += [matrix[i,i]/np.sum(matrix[i,:])]

    res = [acc_tot] + precision + recall
    return np.array(res)
    


# ------------------------------定義recall and Precision-----------------------------------------
def getPrecision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))#N
    TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
    FP=N-TN
    precision = TP / (TP + FP + K.epsilon())#TT/P
    return precision

def getRecall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    P=K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP #FN=P-TP
    recall = TP / (TP + FN + K.epsilon())#TP/(TP+FN)
    return recall
#------------------------------------import image data(gray scale)-----------------------------#
def load_data(dataset_dst):
    with open(dataset_dst, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()

dataset_dst = r"C:\Users\Tina_VI01\Desktop\David\rainbow\dataset\test9_5cls_new\datasets_split.pkl"

(x_train,y_train),(x_test,y_test) = load_data(dataset_dst)
print("x_train.shape = {}".format(x_train.shape))
print("x_test.shape = {}".format(x_test.shape))

imag_w, imag_h = 160, 160
original_dim = imag_w * imag_h
x_train = np.reshape(x_train, [-1, imag_w])
x_test = np.reshape(x_test, [-1, imag_w])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# dim1
X_train = x_train.reshape(-1,imag_h,imag_w,1)
test = x_test.reshape(-1,imag_h,imag_w,1)

# converted to RGB space for training data
x_train3 = np.full((np.size(X_train,0), imag_h, imag_w, 3), 0.0)
for i, s in enumerate(X_train):
    x_train3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) 

# converted to RGB space for testing data
test3 = np.full((np.size(test,0), imag_h, imag_w, 3), 0.0)
for i, s in enumerate(test):
    test3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)


# one-hot encoding
num_cls = 5
Y_train = np_utils.to_categorical(y_train, num_cls)
Y_test = np_utils.to_categorical(y_test, num_cls)
#------------------------------------end of importing data-----------------------------# 


#------------------------------------start building model-----------------------------# 
model = EfficientNetB5(weights = None, input_shape = (imag_h,imag_w,3), include_top=False)

#可控制那些layer可train那些不行(Enet is a pre-trained model)
# for layer in model.layers:
#     layer.trainable=False

ENet_out = model.output
ENet_out = MaxPooling2D(pool_size=(2, 2))(ENet_out)
ENet_out = Flatten()(ENet_out)


Hidden1_in = Dense(1024, activation="relu")(ENet_out)
Hidden1_in = Dropout(0.5)(Hidden1_in)
# predictions = Dense(units = 1, activation="sigmoid")(Hidden1_in)
predictions = Dense(units = num_cls, activation="softmax")(Hidden1_in)
model_f = Model(input = model.input, output = predictions)
# model_f.compile(optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='binary_crossentropy',metrics=["acc",getRecall,getPrecision])
# model_f.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss='categorical_crossentropy', metrics=["acc",getRecall,getPrecision])
model_f.compile(optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy])
# model_f.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])


model_f.load_weights(r"C:\Users\Tina_VI01\Desktop\David\rainbow\B5\test9_res\80 epochs\ENetB5_5cls.h5") #3/11新增
print("> load previous weights successfully...")
#------------------------------------end of building model-----------------------------#



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "To start training model"
    parser.add_argument("-t", "--train", help=help_, action='store_true')
    help_ = "Plot the history"
    parser.add_argument("-p", "--plot", help=help_)
    
    args = parser.parse_args()

    # model_f.summary()
    # plot_model(model_f, to_file='ENetB0.png', show_shapes=True)
    
    #----------------fitting params control------------------#
    epochs = 40
    batch_size = 6
    validation_split=0.1
    shuffle=True
    verbose=1

    w_filename = 'ENetB5_5cls.h5' #跑完最後存檔
    history_filename = 'history_5cls.json' #存每一步的acc
    #----------------fitting params control------------------#

    # Train model
    if args.train: 
        #每一次存最好的一次權重: monitor = loss or val_loss or acc or val_acc
        checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', verbose=verbose,
                                    save_best_only=True, mode='auto', period=1)

        # history = model_f.fit(x_train3, Y_train,
        #         epochs=epochs,
        #         batch_size = batch_size,
        #         validation_data=(test3,Y_test),
        #         shuffle=shuffle,
        #         verbose=verbose,
        #         callbacks=[checkpoint])

        history = model_f.fit(x_train3, Y_train,
                    epochs=epochs,
                    batch_size = batch_size,
                    validation_split=validation_split,
                    shuffle=shuffle,
                    verbose=verbose,
                    callbacks=[checkpoint])

        model_f.save_weights(w_filename)

        with open(history_filename, 'w') as f:
            json.dump(history.history, f)
            # history_df = pd.DataFrame(history.history)
            # history_df[['loss', 'val_loss']].plot()
            # history_df[['acc', 'val_acc']].plot()
            f.close()
        args.plot = history_filename
        
    # load weights
    if args.weights:
        model_f.load_weights(args.weights)

    # ------------------------Prediction----------------------#
    test_predictions = model_f.predict(test3)
    train_predictions = model_f.predict(x_train3)
    # print("test size = ",test_predictions.shape)
    # print("train size = ",train_predictions.shape)
    # print("all test prob = ",test_predictions)

    # select the index with the maximum probability
    y_test_pre = np.argmax(test_predictions,axis = 1)
    y_train_pre = np.argmax(train_predictions,axis = 1)
    # print("type of results = ",type(y_test_pre))
    # print("test result = ", y_test_pre)

    test_matrix = confusion_matrix(y_test, y_test_pre)
    train_matrix = confusion_matrix(y_train, y_train_pre)
    print("test matrix = ", test_matrix)
    print("train matrix = ", train_matrix)

    for md, name in zip([test_matrix,train_matrix],["test_matrix.csv","train_matrix.csv"]):
        df_md = pd.DataFrame(md)
        df_md.to_csv(name)
   
    #Save recall and precision
    a = recall_precision(train_matrix)
    b = recall_precision(test_matrix)
    c = np.vstack((a,b))
    print(c)
    pd.DataFrame(c).to_csv("results_by_matrix.csv")


    # acc = np.sum(results.astype("int32")==y_test.astype("int")/len(y_test)*100)
    # print("the acc = {:.2f} %".format(acc))

    if args.plot:

        with open(args.plot , 'r') as f:
                data = json.load(f)
                history_df = pd.DataFrame(data)

        fig, axes = plt.subplots(1,2,figsize = (12,5))
        
        for x1, x2, y, title , ax in zip(['loss','categorical_accuracy'],['val_loss','val_categorical_accuracy'],['Loss','accuracy(%)'],['Loss Trend', 'Accuracy Trend'],axes.ravel()):
            ax.plot(range(epochs),history_df[x1], label = x1)
            ax.plot(range(epochs),history_df[x2], label = x2)
            ax.set_title(title, fontsize = 20)
            ax.set_xlabel("Nth Epoch", fontsize = 14), ax.set_ylabel(y, fontsize = 14)
            ax.legend(loc = 2)
        
        plt.tight_layout(True)
        fig.savefig("Accuracy.jpg", dpi = 300)

        # fig2, ax = plt.subplots(1,1,figsize=(5,5))
        # ax.set_title("Recall and Precision")
        # for x3 in ['getRecall','getPrecision']:
        #     ax.plot(range(epochs),history_df[x3], label = x3)
        # ax.legend(loc = 2)
        
        # fig2.savefig("Recall_Precision.jpg", dpi = 300)
        # plt.show()
