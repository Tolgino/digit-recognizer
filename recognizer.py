import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
#todo: balanced dataset can be used to predict if it is a digit or not, and then two different models can be used to predict digit and letter separetely.

np.random.seed(42)

def create_model():
    # Load the data
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'archive/emnist-balanced-train.csv')
    train = pd.read_csv(model_path)

    # Extract features and labels
    X_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values

    # Normalize the pixel values
    X_train = X_train / 255.0

    # Reshape to 28x28x1 for the CNN
    X_train = X_train.reshape(-1, 28, 28, 1)

    # One-hot encode the labels
    y_train = to_categorical(y_train, 47)

    # Define the CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(47, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=200)
    
    # Save the model
    
    # Get the directory of the current script
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'new_model.h5')
    model.save(model_path)

    return model

def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'new_model.h5')
    return tf.keras.models.load_model(model_path)

def predict(model, grayscale_array):
    # Create a DataFrame
    df = pd.DataFrame(grayscale_array)

    # Normalize the pixel values
    X_test = df.values / 255.0

    # Reshape the image to match the input shape of the model
    reshaped_image = X_test.reshape(-1, 28, 28, 1)

    # Make predictions using the model
    predictions = model.predict(reshaped_image)

    # Get the predicted character
    predicted_char = np.argmax(predictions)
    
    # Map the character and return
    if predicted_char < 10:
        return str(predicted_char)
    else:
        if predicted_char == 10:
            return 'A'
        elif predicted_char == 11:
            return 'B'
        elif predicted_char == 12:
            return 'C'
        elif predicted_char == 13:
            return 'D'
        elif predicted_char == 14:
            return 'E'
        elif predicted_char == 15:
            return 'F'
        elif predicted_char == 16:
            return 'G'
        elif predicted_char == 17:
            return 'H'
        elif predicted_char == 18:
            return 'I'
        elif predicted_char == 19:
            return 'J'
        elif predicted_char == 20:
            return 'K'
        elif predicted_char == 21:
            return 'L'
        elif predicted_char == 22:
            return 'M'
        elif predicted_char == 23:
            return 'N'
        elif predicted_char == 24:
            return 'O'
        elif predicted_char == 25:
            return 'P'
        elif predicted_char == 26:
            return 'Q'
        elif predicted_char == 27:
            return 'R'
        elif predicted_char == 28:
            return 'S'
        elif predicted_char == 29:
            return 'T'
        elif predicted_char == 30:
            return 'U'
        elif predicted_char == 31:
            return 'V'
        elif predicted_char == 32:
            return 'W'
        elif predicted_char == 33:
            return 'X'
        elif predicted_char == 34:
            return 'Y'
        elif predicted_char == 35:
            return 'Z'
        elif predicted_char == 36:
            return 'A' # normally a
        elif predicted_char == 37:
            return 'B' # normally b
        elif predicted_char == 38:
            return 'D' # normally d
        elif predicted_char == 39:
            return 'E' # normally e
        elif predicted_char == 40:
            return 'F' # normally f
        elif predicted_char == 41:
            return 'G' # normally g
        elif predicted_char == 42:
            return 'H' # normally h
        elif predicted_char == 43:
            return 'N' # normally n
        elif predicted_char == 44:
            return 'Q' # normally q
        elif predicted_char == 45:
            return 'R' # normally r
        elif predicted_char == 46:
            return 'T' # normally t
        else:
            return '-'
