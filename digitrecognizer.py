import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

np.random.seed(42)

def create_model():
    # Load the data
    train = pd.read_csv('./train.csv')

    # Extract features and labels
    X_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values

    # Normalize the pixel values
    X_train = X_train / 255.0

    # Reshape to 28x28x1 for the CNN
    X_train = X_train.reshape(-1, 28, 28, 1)

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)

    # Define the CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=200)
    
    # Save the model
    
    # Get the directory of the current script
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'digit_recognizer_model.h5')
    model.save(model_path)

    return model

def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'digit_recognizer_model.h5')
    return tf.keras.models.load_model(model_path)

def predict_digit(model, grayscale_array):
    # Create a DataFrame
    df = pd.DataFrame(grayscale_array, columns=[f'pixel{i}' for i in range(grayscale_array.shape[1])])

    # Normalize the pixel values
    X_test = df.values / 255.0

    # Reshape the image to match the input shape of the model
    reshaped_image = X_test.reshape(-1, 28, 28, 1)

    # Make predictions using the model
    predictions = model.predict(reshaped_image)

    # Get the predicted digit
    predicted_digit = np.argmax(predictions)

    return predicted_digit