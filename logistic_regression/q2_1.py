import numpy as np
import struct
import matplotlib.pyplot as plt


# Load the Fashion MNIST dataset
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


# Normalize and flatten the images
def preprocess_images(images):
    # Normalize to [0, 1]
    images = images / 255.0
    # Flatten the images
    return images.reshape(images.shape[0], -1)


# One-hot encoding
def one_hot_encoding(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


# Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability adjustment
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Cross-entropy loss with L2 regularization
def compute_loss(y_true, y_pred, weights, l2_reg):
    loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    l2_loss = l2_reg * np.sum(weights**2)
    return loss + l2_loss


# Compute gradient with L2 regularization
def compute_gradient(X, y_true, y_pred, weights, l2_reg):
    m = X.shape[0]
    gradient = -np.dot(X.T, (y_true - y_pred)) / m
    gradient += 2 * l2_reg * weights
    return gradient


# Train the model
def train_logistic_regression(X_train, y_train, X_val, y_val, learning_rate, l2_reg, batch_size, epochs):
    num_samples, num_features = X_train.shape
    num_classes = y_train.shape[1]

    # Improved Weight Initialization (Xavier Initialization)
    weights = np.random.randn(num_features, num_classes) * np.sqrt(1 / num_features)

    for epoch in range(epochs):
        # Shuffle the data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            logits = np.dot(X_batch, weights)
            y_pred = softmax(logits)

            # Compute gradients and update weights
            gradients = compute_gradient(X_batch, y_batch, y_pred, weights, l2_reg)
            weights -= learning_rate * gradients

        # Evaluate on validation data
        logits_val = np.dot(X_val, weights)
        y_pred_val = softmax(logits_val)
        loss = compute_loss(y_val, y_pred_val, weights, l2_reg)
        accuracy = np.mean(np.argmax(y_pred_val, axis=1) == np.argmax(y_val, axis=1))
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    return weights


# Evaluate the model
def evaluate_model(X_test, y_test, weights):
    logits = np.dot(X_test, weights)
    y_pred = softmax(logits)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_labels == y_true_labels)

    # Confusion Matrix
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true_labels, y_pred_labels):
        confusion_matrix[t, p] += 1

    return accuracy, confusion_matrix


# Main function
if __name__ == "__main__":
    # Load dataset
    X_train = load_mnist_images("train-images-idx3-ubyte")
    y_train = load_mnist_labels("train-labels-idx1-ubyte")
    X_test = load_mnist_images("t10k-images-idx3-ubyte")
    y_test = load_mnist_labels("t10k-labels-idx1-ubyte")

    # Preprocess dataset
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)
    y_train = one_hot_encoding(y_train)
    y_test = one_hot_encoding(y_test)

    # Create validation set
    X_val = X_train[:10000]
    y_val = y_train[:10000]
    X_train = X_train[10000:]
    y_train = y_train[10000:]

    # Hyperparameters
    learning_rate = 1e-3  # Increased learning rate
    l2_reg = 1e-3         # Increased regularization
    batch_size = 100      # Reduced batch size
    epochs = 100

    # Train the model
    weights = train_logistic_regression(X_train, y_train, X_val, y_val, learning_rate, l2_reg, batch_size, epochs)

    # Evaluate the model
    test_accuracy, confusion_matrix = evaluate_model(X_test, y_test, weights)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix)
