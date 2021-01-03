import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import mnist
from matplotlib import pyplot as plt

# Started with this tutorial:
# https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# And switch to this one when the first failed
# https://studymachinelearning.com/convolutional-neural-networks-with-keras-in-python/

# Set random seed to follow tutorial at https://elitedatascience.com/keras-tutorial-deep-learning-in-python
np.random.seed(123)

# Load MNIST data into training and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# X is the image, y is the value of the numeral in the image
print("X_train.shape: " + str(X_train.shape)) # expect output is (60000, 28, 28)
print("X_test.shape: " + str(X_test.shape))
print("y_train.shape: " + str(Y_train.shape))
print("y_test.shape: " + str(Y_test.shape))

# Show visualization of data
plt.imshow(X_train[0]/255)
plt.show()

# Show 9 of the mnist training images
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

plt.show()

# Preprocess testing and training data (AKA inputs)
# Specifying a single color channel for input images
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print('1 color channel X_train.shape: {} '.format(X_train.shape))
print('1 color channel X_test.shape: {} '.format(X_test.shape))

# Normalize the pixel values between 0 and 1, but first specify float type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# Preprocess class labels (AKA outputs)
# Convert 1-dimensional class arrays to 10-dimensional class matrices
num_classes=10
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

print("Y_train.shape:" + str(Y_train.shape))
print("Y_test.shape:" + str(Y_test.shape))

# Defining the Convolutional Neural Network model architecture
def prepare_model():
    # Create sequential model
    model = Sequential()
    # Add layers
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

# Fit the model to the training data
model = prepare_model()
model.fit(X_train,Y_train,batch_size=128,epochs=6,verbose=1,validation_data=(X_test, Y_test))

# Plot accuracy of modeling over learning period
acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

# Specify values for first training process graph
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

# Specify values for second training process graph
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Check model accuracy
score = model.evaluate(X_test, Y_test, verbose=0)
print("Test loss is: " + str(score[0]))
print("Test accuracy is: " + str(score[1]))

# Save model
model.save("trained_mnist_digits_model.h5")
