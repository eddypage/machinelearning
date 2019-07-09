
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

# Import digits dataset from sklearn.  Each example is an 8x8 grayscale image of a single digit number
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reshape and normalize inputs
X_train = np.reshape(X_train, (1437, 8, 8, 1))
X_train = X_train/16
X_test = np.reshape(X_test, (360, 8, 8, 1))
X_test = X_test/16

# Convert classification numbers to one-hot representations
y_oh_train = to_categorical(y_train)
y_oh_test = to_categorical(y_test)


def DigitID(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Pads the border of X_input with zeroes to ensure Conv2D layer's output is still 8x8
    X = ZeroPadding2D((1, 1))(X_input)

    # A single Conv2D/BatchNorm/Relu block applied to X_input -> X
    X = Conv2D(16, (3, 3), strides=(1, 1), name = 'conv0')(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # Max pooling to compress activation data
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # Unroll X activations and passing through a single, fully-connected neural layer
    X = Flatten()(X)
    X = Dense(activation='sigmoid', name='fc', units=10)(X)

    # Create Keras model instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='DigitID')

    return model


# Create, compile and fit the model to the training data
digit_model = DigitID((8, 8, 1))
digit_model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
digit_model.fit(X_train, y_oh_train, epochs=40, batch_size=50)

# Check how well the model generalizes using test set.  To optimize hyperparameters, I would use a CV set
train_accuracy = digit_model.evaluate(X_train, y_oh_train, batch_size=32, verbose=1, sample_weight=None)
test_accuracy = digit_model.evaluate(X_test, y_oh_test, batch_size=32, verbose=1, sample_weight=None)

# Print out the results
print()
print ("Training Accuracy = " + str(train_accuracy[1]))
print ("Test Accuracy = " + str(test_accuracy[1]))

"""
Last result:
Training Accuracy = 0.9979123173277662
Test Accuracy = 0.9861111111111112
"""
