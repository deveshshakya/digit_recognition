import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.regularizers import l2
from keras.callbacks import TensorBoard
from keras.utils import np_utils
from util import load_dataset


X_train, y_train, X_test, y_test, classes = load_dataset()

# Normalize image vectors
X_train = X_train / 255
X_test = X_test / 255

# onehot encoding
y_train = np.squeeze(np_utils.to_categorical(y_train))
y_test = np.squeeze(np_utils.to_categorical(y_test))

# creating model
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2()))
model.add(Dense(128, activation='relu', kernel_regularizer=l2()))
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, 
                                         write_graph=True, write_images=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=64, 
          callbacks=[tbCallBack])

model.save_weights('weights.h5')