from __future__ import print_function

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import tensorflow as tf
import numpy as np
import os
import json

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))

seed = 7
np.random.seed(seed)

stop_words = stopwords.words('english')
stemmer = SnowballStemmer("english")


x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')	
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

# initial variables
# vocab_size = len(word2idx)
vocab_size = 3200
batch_size = 16
n_epochs = 100

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)



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
model.add(Dense(1))

# model.summary()

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse'])

callback = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=1,
                    validation_split=0.1,
                    callbacks = [callback])

score = model.evaluate(x_test, y_test, verbose=0)
pred = model.predict(x_test)

train_pred = model.predict(x_train)

score_sort = []
for p, y in zip(pred, y_test):
	score_sort.append( abs(p-y) )
	# print ("pred: %.2f  <-->  %.2f" % (p, y) )

print ("score: %.3f" % score[0])

OUTPUT_PATH = os.path.join('models', 'regression_model.h5')
model.save(OUTPUT_PATH)



