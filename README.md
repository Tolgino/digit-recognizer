# Handwritten Digit Recognizer

This repository contains code for a simple handwritten digit recognizer using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset, a widely used collection of handwritten digits.

# Features

Draw digits on a canvas using your mouse.
Predict the drawn digit(s) by pressing Enter.
Clear the canvas and redraw digits using the Spacebar.
Identify and display predictions for multiple digits drawn simultaneously.
Dependencies

# This code requires the following Python libraries:

pygame
numpy
digitrecognizer (custom module for loading and using the digit recognition model)
OpenCV (for image processing)
Note: You'll need to implement the digitrecognizer.py module to load your trained CNN model. This module should have a function load_model() that returns the loaded model and a function predict_digit(model, image) that takes the model and a preprocessed image (numpy array) as input and returns the predicted digit class.

# How it Works

The code utilizes Pygame to create a graphical interface for drawing digits on a canvas. As you draw, the program tracks your mouse movements and translates them into a NumPy array representing the drawn image.

# Here's a breakdown of the key steps:

Drawing: User interaction with the mouse allows drawing on the canvas.
Prediction: Pressing Enter triggers the prediction process.
The current canvas state is captured and converted to a NumPy array.
OpenCV is used for image pre-processing tasks like converting to grayscale and applying thresholding.
Connected component analysis identifies individual digit drawings within the image.
For each identified component, the image is resized and inverted (as the model might expect a specific format).
The digitrecognizer.py module is used to load the trained model and predict the digit class for each component.
Display: Predicted digits are displayed on the screen alongside instructions for interacting with the program (clear canvas with Spacebar).
This is a basic example of using a pre-trained CNN for digit recognition. You can extend this code by:

Training your own CNN model on a different dataset.
Implementing functionalities for recognizing other characters or shapes.
