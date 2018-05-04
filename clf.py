from __future__ import print_function

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np
import os
import json

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))

seed = 7
np.random.seed(seed)


vocab_size = 3200
# def predict(vocab_size):

x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')	
x_test = np.load('data/x_test.npy')
raw_y_test = np.load('data/y_test.npy')

y_train = to_categorical(y_train)
y_test = to_categorical(raw_y_test)


# initial variables
# vocab_size = len(word2idx)
vocab_size = vocab_size
batch_size = 16
n_epochs = 20

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


s = np.arange(len(x_train))
np.random.shuffle(s)
# self.features = self.features[s]
x_train = x_train[s]
y_train = y_train[s]



# model
model = Sequential()
model.add(Dense(512, init='normal', activation='relu', input_shape=(vocab_size,)))
model.add(Dropout(0.2))
model.add(Dense(256, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy',
   		      optimizer= "adam",
      		  metrics=['acc'])


# # Compile model
# model.compile(loss='categorical_crossentropy', 
# 			  optimizer='adam',
# 			  metrics=[f1])

# callback = EarlyStopping(monitor='val_acc', patience=5)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=1)
                    # validation_split=0.1,
                    # callbacks = [callback])

score = model.evaluate(x_test, y_test, verbose=0)
pred = model.predict_classes(x_test)


# score_sort = []
# for p, y in zip(pred, y_test):
# 	score_sort.append( abs(p-y) )
	# print ("pred: %.2f  <-->  %.2f" % (p, y) )

macro_f1 = f1_score(raw_y_test, pred, average='macro')  
micro_f1 = f1_score(raw_y_test, pred, average='micro')  

print ("=" * 20)
print ("macro_f1: %f" % macro_f1)
print ("micro_f1: %f" % micro_f1)

OUTPUT_PATH = os.path.join('models', 'clf_model.h5')
model.save(OUTPUT_PATH)


# return macro_f1, micro_f1

# np.save('score_sort', np.array(score_sort))
# np.save('pred', pred)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])




