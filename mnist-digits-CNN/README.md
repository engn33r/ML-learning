# Keras MNIST handwritten digits recognition CNN

This project used a Convolutional Neural Network with the MNIST digits training dataset to build a model. The model is then tested against the MNIST test dataset and 9 custom made images that I purposefully made a bit "difficult". The model does indeed make mistakes on the "difficult" custom images, perhaps because they are drawn in a different way than the original dataset.

# Usage
1. Install the dependencies in requirements.txt: `pip install -r requirements.txt`
2. (Optional) Rebuild the model for the MNIST dataset. Note that this may take a few minutes to run: `python build-mnist-model.py`
NOTE: The model building script also opens a few plots and images to give a glimpse of the data being worked with. This step is optional because a model is already provided in this repository.
3. Test the accuracy of the model against a few "difficult" custom inputs: `test-mnist-model.py`

# Credit
Thanks to the following tutorials, among others, for a crash course on keras and ML:
[https://elitedatascience.com/keras-tutorial-deep-learning-in-python](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)
[https://studymachinelearning.com/convolutional-neural-networks-with-keras-in-python/](https://studymachinelearning.com/convolutional-neural-networks-with-keras-in-python/)

