
import tensorflow as tf
import kerastuner as kt
import os
import cv2
import random
import time
import numpy as np
import pickle
import h5py
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout


DATADIR = "faces"
IMG_SIZE = 128
CATEGORIES = ["Akif", "Ali", "Ela", "Fatih", "Furkan"]
num_classes = 5


def create_training_data ():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join (DATADIR, category)
        class_num = CATEGORIES.index (category)
        for img in os.listdir (path):
            try:
                img_array = cv2.imread (os.path.join (path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize (img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append ([new_array, class_num])
            except Exception as e:
                print (e)
                pass
    random.shuffle (training_data)
    return training_data


training_data = create_training_data ()
[training_data, test_data] = train_test_split (training_data)
X = []
y = []

for features, label in training_data:
    X.append (features)
    y.append (label)
X_test = []
y_test = []
for features, label in test_data:
    X_test.append (features)
    y_test.append (label)
X = np.array (X).reshape (-1, IMG_SIZE, IMG_SIZE, 1)
X_test = np.array (X_test).reshape (-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array (y)
y_test = np.array (y_test)

def build_model(hp):
    model=Sequential()
    input_kernel=hp.Choice('input_kernel',[3,5,11])
    input_activation= hp.Choice('input_activation',['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear'])
    input_units=hp.Choice('input_units',[20,32,64,128,256,384,512])
    input_stride=hp.Choice('input_stride',[1,2,3,4])
    #input_padding=hp.Choice('input_padding',['same','valid'])
    model.add(Conv2D(input_units,(input_kernel,input_kernel),strides=(input_stride,input_stride),padding='same',input_shape=(IMG_SIZE,IMG_SIZE,1),activation=input_activation))

    for i in range(hp.Int("Conv_Layer_Groups",min_value=1,max_value=2)):
        for j in range(hp.Int(f'Conv_Layer_Group_{i}_Layers',min_value=1,max_value=4)):
            conv_units=hp.Choice(f'Conv_Group_{i}_Layer_{j}_units',[20,32,64,128,256,384,512])
            conv_kernel=hp.Choice(f'Conv_Group_{i}_Layer_{j}_kernel',[3,5,11])
            conv_activation= hp.Choice(f'Conv_Group_{i}_Layer_{j}_activation',['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear'])
            conv_stride=hp.Choice(f'Conv_Group_{i}_Layer_{j}_stride',[1,2,3,4])
            model.add(Conv2D(conv_units,kernel_size=(conv_kernel,conv_kernel),strides=(conv_stride,conv_stride),activation=conv_activation,padding='same'))
        
        model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    for a in range(hp.Int('Dense_Layers',min_value=1,max_value=2)):
        dense_units=hp.Choice(f'Dense_Layer_{a}_units',[512,1024,2048,4096])        
        dense_activation=hp.Choice(f'Dense_Layer_{a}_activation',['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear'])
        model.add(Dense(units=dense_units,activation=dense_activation))
        drop_rate= hp.Choice(f'Dense_Layer_{a}_dropout_droprate',  [ 0.05, 0.1, 0.15, 0.2,0.25, 0.3, 0.35, 0.40, 0.45, 0.50])
        model.add(Dropout(drop_rate))

    model.add(Dense(5))
    model.add(Activation('softmax'))

    loss='sparse_categorical_crossentropy'
    
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return model
  
LOG_DIR = f"{int (time.time ())}"
tuner=kt.RandomSearch
tuner = kt.RandomSearch(
    build_model,
    objective='val_acc',
    max_trials=2,
    executions_per_trial=1,
    directory=LOG_DIR)

tuner.search(x=X,
             y=y,
             verbose=2, # just slapping this here bc jupyter notebook. The console out was getting messy.
             epochs=1,
             batch_size=64,
             validation_data=(X_test,y_test))

print (tuner.get_best_hyperparameters () [0].values)
print (tuner.results_summary ())
print (tuner.get_best_models () [0].summary ())
model=tuner.get_best_models()[0]
tf.keras.models.save_model(model,'mymodel.h5')
new_model=tf.keras.models.load_model('mymodel.h5')