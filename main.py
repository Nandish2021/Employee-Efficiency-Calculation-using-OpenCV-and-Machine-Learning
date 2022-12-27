import scan
# import motion

import cv2
import numpy as np
import matplotlib.pyplot as plt   # pip install matplotlib

from keras import callbacks

def my_label(image_name):
    name = image_name.split('.')[-3]

    if name=="nandan":
        return np.array([1,0,0])
    elif name=="nandish":
        return np.array([0,1,0])
    elif name=="pradeep":
        return np.array([0,0,1])

import os
from random import shuffle
from tqdm import tqdm
def my_data():
    data = []
    for img in tqdm(os.listdir("data")):
        path=os.path.join("data",img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50,50))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)  
    return data
data = my_data()

train = data[:2400]  
test = data[2400:]
X_train = np.array([i[0] for i in train]).reshape(-1,50,50,1)
print(X_train.shape)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1,50,50,1)
print(X_test.shape)
y_test = [i[1] for i in test]

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

tf.compat.v1.reset_default_graph()
convnet = input_data(shape=[50,50,1])
convnet = conv_2d(convnet, 32, 5, activation='relu')
# 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=1)

# earlystopping = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=2, restore_best_weights=True)

model.fit(X_train, y_train, n_epoch=8, validation_set=(X_test, y_test), show_metric = True, run_id="FRS")#, callbacks=[earlystopping])



'''
def data_for_visualization():
    Vdata = []
    for img in tqdm(os.listdir("Images for visualization")):
      try:
        path = os.path.join("Images for visualization", img)
        img_num = img.split('.')[0] 
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50,50))
        Vdata.append([np.array(img_data), img_num])
      except:
        continue
    shuffle(Vdata)
    return Vdata
'''

def data_for_visualization():
    Vdata = []
    img_lst = scan.generate_dataset()
    i=0
    for img in img_lst:
        try:
            img_data = cv2.resize(img, (50,50))
            Vdata.append([np.array(img_data), i])
            i+=1
        except:
            continue
    return Vdata


while True:
    # video = cv2.VideoCapture(0)
    motion.mot()
    # print(f)
    # key = cv2.waitKey(100)
    Vdata = data_for_visualization()

    fig = plt.figure(figsize=(20,20))
    for num, data in enumerate(Vdata[:20]):
        img_data = data[0]
        y = fig.add_subplot(5,5, num+1)
        image = img_data
        data = img_data.reshape(50,50,1)
        model_out = model.predict([data])[0]

        print(model_out)
        
        if np.argmax(model_out) == 0 and model_out[0]>0.6:
            my_label = 'Nandan'
        elif np.argmax(model_out) == 1 and model_out[1]>0.6:
            my_label = 'Nandish'
        elif np.argmax(model_out) == 2 and model_out[2]>0.6:
            my_label = 'Pradeep'
        else:
            my_label = 'Unknown'
            
        y.imshow(image, cmap='gray')
        plt.title(my_label)
        
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()

''''''