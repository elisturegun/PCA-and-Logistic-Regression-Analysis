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
    images = images / 255.0  # Normalize to [0, 1]
    return images.reshape(images.shape[0], -1)  # Flatten the images


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


# Training function with epoch-wise logging
def train_logistic_regression_with_logging(
    X_train, y_train, X_val, y_val, initial_lr, l2_reg, batch_size, epochs, decay_rate=0.01, momentum=0.9, init="xavier"
):
    num_samples, num_features = X_train.shape
    num_classes = y_train.shape[1]

    # Weight initialization
    if init == "zero":
        weights = np.zeros((num_features, num_classes))
    elif init == "uniform":
        weights = np.random.uniform(-0.01, 0.01, (num_features, num_classes))
    elif init == "normal":
        weights = np.random.randn(num_features, num_classes)
    else:  # Default Xavier initialization
        weights = np.random.randn(num_features, num_classes) * np.sqrt(1 / num_features)

    velocity = np.zeros_like(weights)  # Initialize velocity for momentum

    accuracies = []  # Store validation accuracy for plotting

    for epoch in range(epochs):
        # Shuffle the data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Decay the learning rate
        learning_rate = initial_lr / (1 + decay_rate * epoch)

        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            logits = np.dot(X_batch, weights)
            y_pred = softmax(logits)

            # Compute gradients
            gradients = compute_gradient(X_batch, y_batch, y_pred, weights, l2_reg)

            # Update weights with momentum
            velocity = momentum * velocity - learning_rate * gradients
            weights += velocity

        # Evaluate on validation data
        logits_val = np.dot(X_val, weights)
        y_pred_val = softmax(logits_val)
        accuracy = np.mean(np.argmax(y_pred_val, axis=1) == np.argmax(y_val, axis=1))
        accuracies.append(accuracy)

        # Print epoch-wise accuracy
        print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {accuracy:.4f}")

    return accuracies


# Run experiments for a single hyperparameter
def run_experiments(hyperparameter_name, values, X_train, y_train, X_val, y_val):
    results = {}
    for value in values:
        print(f"\nRunning {hyperparameter_name} experiment with value: {value}")
        if hyperparameter_name == "batch_size":
            accuracies = train_logistic_regression_with_logging(
                X_train, y_train, X_val, y_val, initial_lr=1e-3, l2_reg=1e-3, batch_size=value, epochs=100
            )
        elif hyperparameter_name == "init":
            accuracies = train_logistic_regression_with_logging(
                X_train, y_train, X_val, y_val, initial_lr=1e-3, l2_reg=1e-3, batch_size=100, epochs=100, init=value
            )
        elif hyperparameter_name == "learning_rate":
            accuracies = train_logistic_regression_with_logging(
                X_train, y_train, X_val, y_val, initial_lr=value, l2_reg=1e-3, batch_size=100, epochs=100
            )
        elif hyperparameter_name == "l2_reg":
            accuracies = train_logistic_regression_with_logging(
                X_train, y_train, X_val, y_val, initial_lr=1e-3, l2_reg=value, batch_size=100, epochs=100
            )
        results[value] = accuracies

    return results


# Plot the results
def plot_results(results, hyperparameter_name):
    plt.figure(figsize=(10, 6))
    for value, accuracies in results.items():
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=f"{hyperparameter_name}={value}")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Validation Accuracy vs Epochs for {hyperparameter_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function for hyperparameter experiments
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

    # Experiment 1: Batch size
    batch_sizes = [1, 64, 3000]
    batch_results = run_experiments("batch_size", batch_sizes, X_train, y_train, X_val, y_val)
    plot_results(batch_results, "Batch Size")

    # Experiment 2: Weight initialization
    inits = ["zero", "uniform", "normal"]
    init_results = run_experiments("init", inits, X_train, y_train, X_val, y_val)
    plot_results(init_results, "Weight Initialization")

    # Experiment 3: Learning rate
    learning_rates = [0.01, 1e-3, 1e-4, 1e-5]
    lr_results = run_experiments("learning_rate", learning_rates, X_train, y_train, X_val, y_val)
    plot_results(lr_results, "Learning Rate")

    # Experiment 4: Regularization coefficient
    l2_regs = [1e-2, 1e-4, 1e-9]
    l2_results = run_experiments("l2_reg", l2_regs, X_train, y_train, X_val, y_val)
    plot_results(l2_results, "Regularization Coefficient")
