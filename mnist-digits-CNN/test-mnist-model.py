import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.datasets import mnist
from matplotlib import pyplot as plt

# Function to convert an image into an array that the computer can operate on
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode = "grayscale", target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# Load saved model and view summary
model = load_model('trained_mnist_digits_model.h5')
model.summary()	

# Load the original mnist data for a quick accuracy check
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Check the accuracy of the imported model, for verification purposes
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

# See if model can properly identify custom-made inputs
for customnumber in range(1,10):
    img = load_image('custom-num' + str(customnumber) + '.jpg')
    digit = np.argmax(model.predict(img), axis=-1)
    print("Predicted digit: ",digit[0])
    print("Actual digit: ", customnumber)

