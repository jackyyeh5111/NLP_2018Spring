import pickle
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.layers import LSTM, Bidirectional
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
import tensorflow as tf
import numpy as np
from optparse import OptionParser
import json
from util import match_relation

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))

# fix random seed for reproducibility
np.random.seed(7)

# opts
op = OptionParser()
op.add_option("--process_name",
              dest="process_name", type=str,
              help="process name")
op.add_option("--cell",
              dest="cell", type=str)

(opts, args) = op.parse_args()
if not opts.process_name:  op.error('process_name is not given')
if not opts.cell:  op.error('cell is not given')



RESULT_PATH = "./results/proposed_answer_" + opts.process_name +  ".txt"
MODEL_PATH = "./model/model_" + opts.process_name + ".h5"


# x_train = np.load('./data/x_train.npy')
x_train = np.load('./data/elmo_x_train.npy')
y_train = np.load('./data/y_train.npy')
x_test = np.load('./data/elmo_x_test.npy')

# Variable
n_units = 256    # hidden LSTM units
n_classes = 19 
batch_size = 128 # Size of each batch
n_epochs = 100
embed_dim = 1024
time_steps = x_train.shape[1]


y_train = to_categorical(y_train, n_classes)


################################ model ################################
model = Sequential()

if opts.cell == "lstm":
  model.add(LSTM(n_units, input_shape=(time_steps, embed_dim)))
  model.add(Dropout(0.2))

elif opts.cell == "bi-lstm":
  model.add(Bidirectional(LSTM(n_units), input_shape=(time_steps, embed_dim)))
  model.add(Dropout(0.2))

elif opts.cell == "multi-lstm":
  model.add(LSTM(n_units, return_sequences=True, input_shape=(time_steps, embed_dim)))
  model.add(Dropout(0.2))
  model.add(LSTM(n_units))
  model.add(Dropout(0.2))


model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


callback = EarlyStopping(monitor='val_acc', patience=8)

model.fit(x_train, y_train, 
          batch_size=batch_size,
          epochs=n_epochs,
          verbose=1,
          validation_split=0.1,
          callbacks = [callback])

model.save(MODEL_PATH)

# model = load_model(MODEL_PATH)

pred_classes = model.predict_classes(x_test)

print (pred_classes)

pointer = 8001
with open(RESULT_PATH, 'w') as f:
    for i, pred_class in enumerate(pred_classes):
        relation = match_relation(pred_class)
        f.write("%d\t%s\n" % (i+pointer, relation))





