import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#  question 1.1
def load_images(folder):
    """
    Load images from the folder, resize to 64x64, and convert them into PyTorch tensors.
    Returns a tensor of shape (10000, 4096, 3) for 10,000 images.
    """
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = Image.open(os.path.join(folder, filename))
            img_resized = img.resize((64, 64), Image.BILINEAR)
            img_array = np.asarray(img_resized, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            images.append(img_array)

    images = np.stack(images)  # Shape: (10000, 64, 64, 3)
    images = torch.tensor(images).permute(0, 3, 1, 2)  # Shape: (10000, 3, 64, 64)
    print(f"Loaded {images.shape[0]} images with shape {images.shape}")
    return images


def preprocess_images(images, device):
    """
    Flatten each image and move the tensor to the specified device.
    Returns tensors for each color channel: Red, Green, and Blue.
    """
    # Flatten images (3 channels, 64x64 -> 4096)
    images_flat = images.view(images.size(0), 3, -1)  # Shape: (10000, 3, 4096)
    # Split channels and move to GPU
    red_channel = images_flat[:, 0, :].to(device)
    green_channel = images_flat[:, 1, :].to(device)
    blue_channel = images_flat[:, 2, :].to(device)
    return red_channel, green_channel, blue_channel


def compute_pca(X):
    """
    Perform PCA on the given data tensor X (shape: [10000, 4096]).
    Returns eigenvalues, eigenvectors, and proportion of variance explained (PVE).
    """
    # Center the data
    mean = torch.mean(X, dim=0, keepdim=True)
    X_centered = X - mean

    # Compute covariance matrix
    covariance_matrix = torch.mm(X_centered.T, X_centered) / (X_centered.size(0) - 1)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)  # Use GPU
    eigenvalues, eigenvectors = eigenvalues.flip(0), eigenvectors.flip(1)  # Sort descending

    # Compute Proportion of Variance Explained (PVE)
    total_variance = torch.sum(eigenvalues)
    pve = eigenvalues / total_variance
    return eigenvalues, eigenvectors, pve

def visualize_pve(pve, channel_name):
    """
    Visualize the PVE and cumulative PVE for a given channel.
    """
    # Convert PVE to numpy for plotting
    pve = pve.cpu().numpy()

    # Compute cumulative PVE
    cumulative_pve = np.cumsum(pve)

    # Plot PVE
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), pve[:10], label="PVE (Top 10 Components)", alpha=0.7)
    plt.plot(range(1, len(cumulative_pve) + 1), cumulative_pve, label="Cumulative PVE", color="purple", marker="o")
    plt.axhline(y=0.7, color="green", linestyle="--", label="70% Threshold")

    # Add labels, legend, and title
    plt.xlabel("Principal Component Index")
    plt.ylabel("Proportion of Variance Explained (PVE)")
    plt.title(f"PCA Analysis - {channel_name} Channel")
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_channel(channel_name, channel_data):
    """
    Perform PCA analysis for a given color channel and report results.
    """
    print(f"\nAnalyzing {channel_name} channel...")
    eigenvalues, eigenvectors, pve = compute_pca(channel_data)

    # Report PVE for the first 10 principal components
    print(f"Top 10 PVE for {channel_name} channel: {pve[:10].cpu().numpy()}")
    print(f"Total PVE for top 10 components: {torch.sum(pve[:10]).item()}")

    # Find minimum number of components to achieve 70% PVE
    cumulative_pve = torch.cumsum(pve, dim=0)
    min_components = torch.nonzero(cumulative_pve >= 0.7)[0].item() + 1
    print(f"Minimum components for 70% PVE: {min_components}")

     # Visualize PVE
    visualize_pve(pve, channel_name)

# question 1.2
def reshape_and_normalize(eigenvectors, channel_name):
    """
    Reshape and normalize the first 10 principal components.
    Returns a list of reshaped and normalized components.
    """
    reshaped_components = []
    for i in range(10):  # First 10 components
        component = eigenvectors[:, i].reshape(64, 64)  # Reshape to 64x64
        # Min-Max scaling
        component_min = torch.min(component)
        component_max = torch.max(component)
        component_normalized = (component - component_min) / (component_max - component_min)
        reshaped_components.append(component_normalized.cpu().numpy())  # Convert to numpy for plotting
    print(f"Reshaped and normalized first 10 components for {channel_name} channel.")
    return reshaped_components

def create_rgb_images(red_components, green_components, blue_components):
    """
    Stack the corresponding Red, Green, and Blue components to form 10 RGB images.
    Returns a list of 10 RGB images.
    """
    rgb_images = []
    for i in range(10):
        rgb_image = np.stack([red_components[i], green_components[i], blue_components[i]], axis=-1)  # Stack along the last dimension
        rgb_images.append(rgb_image)
    print("Created 10 RGB images.")
    return rgb_images

def display_rgb_images(rgb_images):
    """
    Display the 10 RGB images in a grid.
    """
    plt.figure(figsize=(15, 10))
    for i, rgb_image in enumerate(rgb_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(rgb_image)
        plt.axis("off")
        plt.title(f"PC {i + 1}")
    plt.suptitle("First 10 Principal Components as RGB Images")
    plt.show()

def visualize_principal_components(eigenvectors_red, eigenvectors_green, eigenvectors_blue):
    """
    Process and visualize the first 10 principal components for all channels.
    """
    # Reshape and normalize
    red_components = reshape_and_normalize(eigenvectors_red, "Red")
    green_components = reshape_and_normalize(eigenvectors_green, "Green")
    blue_components = reshape_and_normalize(eigenvectors_blue, "Blue")
    
    # Create RGB images
    rgb_images = create_rgb_images(red_components, green_components, blue_components)
    
    # Display RGB images
    display_rgb_images(rgb_images)

#  question 1.3
def reconstruct_image(image, mean, eigenvectors, k, channel_name):
    """
    Reconstruct an image using the first k principal components.
    Arguments:
        image: The original image (1D vector).
        mean: The mean vector subtracted during PCA computation.
        eigenvectors: The eigenvectors obtained from PCA.
        k: The number of principal components to use for reconstruction.
        channel_name: The color channel being processed.
    Returns:
        Reconstructed image as a 2D array.
    """
    # Center the image by subtracting the mean
    image_centered = image - mean

    # Project the image onto the first k eigenvectors
    projection = torch.mm(image_centered.view(1, -1), eigenvectors[:, :k])

    # Reconstruct the image using the projection
    reconstructed = torch.mm(projection, eigenvectors[:, :k].T)

    # Add the mean back to the reconstructed image
    reconstructed += mean

    # Reshape the 1D image back to 64x64
    reconstructed_image = reconstructed.view(64, 64)
    
    print(f"Reconstructed {channel_name} channel with k={k} components.")
    return reconstructed_image.cpu().numpy()  # Convert to numpy for visualization

def reconstruct_and_visualize(image_path, mean_red, mean_green, mean_blue, eigenvectors_red, eigenvectors_green, eigenvectors_blue):
    """
    Reconstruct the image for different k values and visualize results.
    """
    # Load and preprocess the image
    image = Image.open(image_path).resize((64, 64), Image.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0  # Normalize
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).view(3, -1)  # Shape: (3, 4096)

    # Extract color channels
    red_channel = image_tensor[0, :].to(device)
    green_channel = image_tensor[1, :].to(device)
    blue_channel = image_tensor[2, :].to(device)

    # Define k values
    k_values = [1, 50, 250, 500, 1000, 4096]

    # Perform reconstruction for each k value
    reconstructed_images = []
    for k in k_values:
        print(f"\nReconstructing with k={k}...")
        red_reconstructed = reconstruct_image(red_channel, mean_red, eigenvectors_red, k, "Red")
        green_reconstructed = reconstruct_image(green_channel, mean_green, eigenvectors_green, k, "Green")
        blue_reconstructed = reconstruct_image(blue_channel, mean_blue, eigenvectors_blue, k, "Blue")

        # Stack RGB channels and reshape to original image shape
        reconstructed_image = np.stack([red_reconstructed, green_reconstructed, blue_reconstructed], axis=-1)
        reconstructed_images.append(reconstructed_image)

    # Visualize reconstructed images
    visualize_reconstructed_images(reconstructed_images, k_values)

def visualize_reconstructed_images(reconstructed_images, k_values):
    """
    Visualize reconstructed images for different k values.
    """
    plt.figure(figsize=(20, 10))
    for i, (image, k) in enumerate(zip(reconstructed_images, k_values)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"k={k}")
    plt.suptitle("Reconstructed Images with Varying Number of Principal Components")
    plt.show()

def main(folder, device):
    # Load images
    images = load_images(folder)

    # Preprocess images
    red_channel, green_channel, blue_channel = preprocess_images(images, device)

    # Analyze each channel
    for channel_name, channel_data in zip(['Red', 'Green', 'Blue'], [red_channel, green_channel, blue_channel]):
        analyze_channel(channel_name, channel_data)

     # Perform PCA and visualize results
    eigenvectors_red, eigenvectors_green, eigenvectors_blue = None, None, None
    for channel_name, channel_data in zip(['Red', 'Green', 'Blue'], [red_channel, green_channel, blue_channel]):
        print(f"\nAnalyzing {channel_name} channel...")
        eigenvalues, eigenvectors, pve = compute_pca(channel_data)
        if channel_name == "Red":
            eigenvectors_red = eigenvectors
        elif channel_name == "Green":
            eigenvectors_green = eigenvectors
        elif channel_name == "Blue":
            eigenvectors_blue = eigenvectors

    # Visualize the first 10 principal components as RGB images
    visualize_principal_components(eigenvectors_red, eigenvectors_green, eigenvectors_blue)

    # Calculate means
    mean_red = torch.mean(red_channel, dim=0)
    mean_green = torch.mean(green_channel, dim=0)
    mean_blue = torch.mean(blue_channel, dim=0)

    # Reconstruct and visualize image 9577.png
    reconstruct_and_visualize("./data/StyleGAN_fake/resized_images/image_9577.png", mean_red, mean_green, mean_blue, eigenvectors_red, eigenvectors_green, eigenvectors_blue)



if __name__ == "__main__":

    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the dataset folder path relative to the script's location
    dataset_folder = os.path.join(script_dir, "./data/StyleGAN_fake/resized_images")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main(dataset_folder, device)

