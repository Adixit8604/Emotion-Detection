import numpy as np
import pandas as pd
import sklearn.model_selection
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Load the CSV file
data = pd.read_csv("Data/train.csv")

# Parse the data
pixels = data["pixels"].values
labels = data["emotion"].values

# Convert pixel values to images and normalize them
images = []
for pixel_sequence in pixels:
    image = [int(pixel) for pixel in pixel_sequence.split()]
    image = np.array(image).reshape(48, 48)  # Assuming images are 48x48 pixels
    image = image / 255.0  # Normalize pixel values
    images.append(image)
images = np.array(images)


# Convert labels to categorical format
labels = to_categorical(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(images, labels, test_size=0.33, random_state=101)

# Load the saved model
model = load_model("emotion_detection_model.h5")

# Train the model
history = model.fit(X_train, y_train, batch_size=64, epochs=15, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Save the trained model
model.save("emotion_detection_model_trained.h5")
