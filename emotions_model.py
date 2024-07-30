# Importing the dependencies

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Defining the directory and emotions
data_dir = 'fer2013'
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Loading the FER-2013 dataset
def load_fer2013(data_dir, emotions):
    faces = []
    labels = []
    for idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
            img = img_to_array(img)
            faces.append(img)
            labels.append(idx)
    faces = np.array(faces)
    labels = to_categorical(labels, num_classes=len(emotions))
    return faces, labels

# Preprocessing the dataset
faces, labels = load_fer2013(data_dir, emotions)
faces = faces / 255.0  # Normalizing the pixel values

# Spliting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

# Building the CNN model
def build_emotion_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(emotions), activation='softmax'))
    return model

# Compiling the model
emotion_model = build_emotion_model()
emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
emotion_model.fit(x_train, y_train, epochs=60, batch_size=64, validation_data=(x_test, y_test))

# Saving the trained model
emotion_model.save('emotion_model.h5')

print("Model trained and saved as emotion_model.h5")
