import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(48,48, 1)),
    Activation('elu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same'),
    Activation('elu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), padding='same'),
    Activation('elu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same'),
    Activation('elu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2),padding='same'),
    Dropout(0.2),
    Conv2D(128, (3, 3), padding='same'),
    Activation('elu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same'),
    Activation('elu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2),padding='same'),
    Dropout(0.2),
    Conv2D(256, (3, 3), padding='same'),
    Activation('elu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same'),
    Activation('elu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2),padding='same'),
    Dropout(0.2),
    Flatten(),
    Dense(128),
    Activation('elu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64),
    Activation('elu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32),
    Activation('elu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16),
    Activation('elu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(7),
    Activation('softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Save the model to disk in .h5 format
# Save the model in .keras format
model.save("emotion_detection_model.h5")


# Print model summary
model.summary()
