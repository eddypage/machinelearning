
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


import keras.backend as K
K.set_image_data_format('channels_last')


X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = np.reshape(X_train, (1437, 8, 8, 1))
X_train = X_train/16
X_test = np.reshape(X_test, (360, 8, 8, 1))
X_test = X_test/16


y_oh_train = to_categorical(y_train)
y_oh_test = to_categorical(y_test)


def DigitID(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((1, 1))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(16, (3, 3), strides=(1, 1), name = 'conv0')(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(activation='sigmoid', name='fc', units=10)(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='DigitID')

    return model


print(y_oh_train.shape)

digit_model = DigitID((8, 8, 1))
digit_model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
digit_model.fit(X_train, y_oh_train, epochs=40, batch_size=50)

prediction = digit_model.evaluate(X_test, y_oh_test, batch_size=32, verbose=1, sample_weight=None)

print()
print ("Loss = " + str(prediction[0]))
print ("Test Accuracy = " + str(prediction[1]))
