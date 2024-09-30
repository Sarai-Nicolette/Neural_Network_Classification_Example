"""
*************************************************************************
**             Neural Network Classification Example                   **
**                                                                     **
** This program demonstrates a classification problem solved by        **
** utilizing neural networks, tensorflow, and other related packages.  **
** Created by: Dr. Sarai Sherfield                                     **
** Last modified on: 9/25/2024 by Sarai Sherfield                      **
*************************************************************************
"""

# Libraries
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Functions
def plot_decision_boundary(model, x_data, y_data):
    """
    Plots the decision boundary created by a model predicting on X.
    This function was inspired by two resources:
    1. https://cs231n.github.io/neural-networks-case-study/
    2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
    """
    # Define the axis boundaries of a plot and create a meshgrid
    x_min, x_max = x_data[:, 0].min() - 0.1, x_data[:, 0].max() + 0.1
    y_min, y_max = x_data[:, 1].min() - 0.1, x_data[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X value (we're going to make predictions on these)
    x_in = np.c_[xx.ravel(), yy.ravel()]  # stack 2D arrays together

    # Make predictions
    y_pred = model.predict(x_in)

    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        # We have to reshape our prediction to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Make 500 samples
n_samples = 1200

# Create circles (Small and large circles are creates
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Uncomment below to visualize circles
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.title("Random Circles")
# plt.show()

# Split into training and testing datasets
X_train, y_train = X[:850], y[:850]
X_test, y_test = X[850:], y[850:]

# Set the random seed
tf.random.set_seed(42)

# Create model
model_circles = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

# Compile model
model_circles.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      metrics=["accuracy"])

# Fit the model
history = model_circles.fit(X_train, y_train, epochs=45, verbose=0)

# Evaluate the model
model_circles.evaluate(X_test, y_test)

# Visualize the model in training and in testing
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_circles, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_circles, X_test, y_test)
plt.show()
