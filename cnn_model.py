# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
 

 # Import Keras layers and models for building the CNN
layers = keras.layers
models = keras.models  

# Import NumPy for numerical operations
import numpy as np  

# Import Matplotlib for visualization
import matplotlib.pyplot as plt  
from tensorflow.keras.callbacks import EarlyStopping
# Load the MNIST dataset (handwritten digits)
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to include a single channel (grayscale images)
x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32")   # Training data
x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32")  # Testing data

# Normalize the pixel values to range [0,1] to improve model efficiency
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the CNN model architecture
model = keras.models.Sequential([
    # First convolutional layer (32 filters, 3x3 kernel, ReLU activation)
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),  # Max pooling to reduce spatial dimensions

    # Second convolutional layer (64 filters, 3x3 kernel)
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),  # Another max pooling layer
    keras.layers.Dropout(0.25),
    # Thiryers.Dropout(0.25),d convolutional layer (64 filters, 3x3 kernel)
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Dropout(0.25),

    # Flatten the 2D feature maps into a 1D vector
    keras.layers.Flatten(),

    # Fully connected layer with 64 neurons and ReLU activation
    keras.layers.Dense(64, activation='relu'),

    # Output layer with 10 neurons (one for each digit 0-9) using softmax activation
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model (optimizer, loss function, and evaluation metric)
model.compile(optimizer='adam',  # Adam optimizer for efficient training
              loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
              metrics=['accuracy'])  # Measure accuracy during training

# Train the model using the training dataset
model.fit(x_train, y_train, epochs=10, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])  # Train for 5 epochs

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)  
print(f"Test accuracy: {test_acc:.4f}")  # Print final test accuracy

# Display a sample test image with its predicted label
index = 0  # Change this index to test different images
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')  # Show the selected test image

# Make a prediction using the trained model
prediction = model.predict(np.expand_dims(x_test[index], axis= 0))  # Predict for this image
predicted_label = np.argmax(prediction)  # Get the predicted label

# Display the predicted label on the image
plt.title(f"Predicted Label: {predicted_label}")  # Display the predicted label
plt.show()

# Print the result in the console
print(f"Actual Label: {y_test[index]}, Predicted Label: {predicted_label}")