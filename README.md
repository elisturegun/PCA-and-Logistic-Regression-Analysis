# **PCA-FAKE-FACES-ANALYSIS AND FASHION MNIST LOGISTIC REGRESSION PROJECT**

This project implements solutions to the following tasks:

1. **PCA Analysis**: Performing Principal Component Analysis (PCA) on StyleGAN-generated fake face images to analyze their principal components.
2. **Logistic Regression**: Training and evaluating a logistic regression model on the Fashion MNIST dataset, including experiments with hyperparameter tuning.

## **Installation**

### **Prerequisites**
- Python 3.8 or higher
- Required Python libraries:
  - `numpy`
  - `torch`
  - `Pillow`
  - `matplotlib`


## **Setup**
1. Clone or download the project files to your local machine.

2. Navigate to the project directory:
   cd PCA-and-Logistic-Regression-Analysis

3. Install the required dependencies
    pip install -r requirements.txt

### **Dataset Preparation**
**Fashion MNIST Dataset**

    Link: [Fashion MNIST Classification](https://www.kaggle.com/code/ohwhykate/fashion-mnist-classification)
    Description: Contains labeled grayscale images of clothing items for classification tasks.

**StyleGAN Fake Face Dataset**

    Link: https://drive.google.com/file/d/1NLfnWvmIlP9dvQugOugxAKXgzI2qK_i7/view 
    Description: A collection of 10,000 artificially generated fake face images resized to 64x64 pixels.

**File Placement**
Ensure that these datasets are downloaded and placed in the following locations:

**Fashion MNIST Files**

    logistic_regression/t10k-images-idx3-ubyte (Test images)
    logistic_regression/t10k-labels-idx1-ubyte (Test labels)
    logistic_regression/train-images-idx3-ubyte (Training images)
    logistic_regression/train-labels-idx1-ubyte (Training labels)

**StyleGAN Fake Face Images**

    Place all .png images in:
    pca_analysis/data/StyleGAN_fake/resized_images/

### **Running the Project**
The project is executed via the main.py driver script. It automates the tasks for both PCA analysis and logistic regression.
Run the script:
    python main.py

### **Functionality**
**1. PCA Analysis**
Located in the pca_analysis folder, the PCA analysis includes:

    Task 1: Calculation of the top 10 principal components and their proportions of variance explained.

    Task 2: Visualization of RGB images of the first 10 principal components.

    Task 3: Reconstruction of an image using different numbers of principal components.

    Outputs for these tasks are stored in .png format in the pca_analysis folder.

**2. Logistic Regression**
Located in the logistic_regression folder, this section includes:

Task 2.1 - 2.5: Training the logistic regression model on the Fashion MNIST dataset, performing hyperparameter experiments (batch size, learning rate, weight initialization, regularization), and visualizing results like confusion matrices and weight vectors.
Outputs for logistic regression include .png visualizations and performance metrics.

    Task 1: Train the default logistic regression model and display the test accuracy and confusion matrix.

    Task 2: Perform separate experiments on batch size, weight initialization, learning rate, and regularization coefficient, and compare their performances with accuracy graphs.

    Task 3: Select the best hyperparameters and initialization technique, train the optimal model, and display the test accuracy and confusion matrix.

    Task 4: Visualize the finalized weight vectors of the best model as images and comment on their representation.

    Task 5: Calculate precision, recall, F1 score, and F2 score for each class using the best model and analyze results using the confusion matrix and weight images.

    Outputs for these tasks are stored in .png format in the logistic_regression folder.
