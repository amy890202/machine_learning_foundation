# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 19:10:26 2021

@author: user
"""
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras.utils import np_utils, to_categorical
import matplotlib.pyplot as plt


(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_train_norm = x_train/255
y_trainonehot = np_utils.to_categorical(y_train)

model = Sequential()
model.add(Conv2D(filters=8,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=10,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'])
train_history = model.fit(x=x_train_norm ,y= y_trainonehot,epochs=50,batch_size=256,verbose=2)
    
#%%

index0 = np.where(y_train==0)[0]#把數字0跟數字1的資料都先拿出來
index1 = np.where(y_train==1)[0]
index_all = np.append(index0,index1)
y_binary = y_train[index_all]
x_binary = x_train[index_all]
x_binary = x_binary.reshape(-1,28*28).astype('float32')/255
sn = x_binary.shape[0]
idx = np.random.permutation(sn)
x_binary = x_binary[idx,:]
y_binary = y_binary[idx]



model_binary = Sequential()
model_binary.add(Dense(50,input_dim=784,activation = 'sigmoid'))#relu
model_binary.add(Dense(20,input_dim=784,activation = 'relu'))#relu
model_binary.add(Dense(20,input_dim=784,activation = 'relu'))#relu
model_binary.add(Dense(10,input_dim=784,activation = 'relu'))#relu
model_binary.add(Dense(10,input_dim=784,activation = 'relu'))#relu
model_binary.add(Dense(1,activation='sigmoid'))
model_binary.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
training_binary = model_binary.fit(x=x_binary,y=y_binary,validation_split=0.3,epochs=10,batch_size=128,verbose=2)


model_binary.summary()

#%%
x_train = x_train.reshape(-1,28*28).astype('float32')/255#把28*28的照片拉成1*784
y_trainonehot = np_utils.to_categorical(y_train)
model_categorical = Sequential()
model_categorical.add(Dense(28,input_dim = 784,activation ='relu'))
model_categorical.add(Dense(24,activation ='relu'))
model_categorical.add(Dense(20,activation ='relu'))
model_categorical.add(Dense(16,activation ='relu'))
model_categorical.add(Dense(10,activation ='softmax'))#10個output summation會是1
model_categorical.summary()
model_categorical.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
training_categorical = model_categorical.fit(x=x_train,y=y_trainonehot,validation_split=0.3,epochs=100,batch_size=1024,verbose=2)
#batch_size越大越平穩、更新越快、比較容易忽略個體差異(因為是一個群體平均去看)
#%%
plt.plot(training_binary.history['loss'],label='loss')
plt.plot(training_binary.history['val_loss'],label='val_loss')
plt.legend()
plt.show()
plt.plot(training_binary.history['accuracy'],label='acc')
plt.plot(training_binary.history['val_accuracy'],label='val_acc')
plt.legend()
plt.show()

