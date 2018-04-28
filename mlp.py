from __future__ import print_function

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.decomposition import TruncatedSVD
import numpy as np
import json

seed = 7
np.random.seed(seed)

with open("data/word2idx.json", 'r') as f:
    word2idx = json.load(f)

with open("data/idx2word.json", 'r') as f:
    idx2word = json.load(f)

x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')	
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

# initial variables
vocab_size = len(word2idx)
batch_size = 16
n_epochs = 100


# bag of words
def BOW_encode(raw_sequence):

	BOW_vectors = np.zeros([len(raw_sequence), vocab_size], dtype=np.int32)
	for i, seq in enumerate(raw_sequence):
		one_hot_vector = np.zeros([vocab_size], dtype=np.int32)
	
		for idx in seq:
			one_hot_vector[idx] += 1

		BOW_vectors[i] = one_hot_vector

	return BOW_vectors


# bag of words 
x_train = BOW_encode(x_train)
x_train = np.array(x_train)


x_test = BOW_encode(x_test)
x_test = np.array(x_test)



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

model.summary()


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

score_sort = []
for p, y in zip(pred, y_test):
	score_sort.append( abs(p-y) )
	print ("pred: %.2f  <-->  %.2f" % (p, y) )

print ("score: %.3f" % score[0])
np.save('score_sort', np.array(score_sort))
np.save('pred', pred)

np.save('score_sort', score_sort)
np.save('pred', pred)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])




