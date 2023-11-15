# -Machine-learning-Cancer-cell-classification-using-Scikit-learn
## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement:



## Dataset:



## Neural Network Model





## DESIGN STEPS

### Step 1:
We begin by importing the necessary Python libraries, including TensorFlow for deep learning, data preprocessing tools, and visualization libraries.

### Step 2:
To leverage the power of GPU acceleration, we configure TensorFlow to allow GPU processing, which can significantly speed up model training.

### Step 3:
We load the dataset, consisting of cell images, and check their dimensions. Understanding the image dimensions is crucial for setting up the neural network architecture.

### Step 4:
We create an image generator that performs data augmentation, including rotation, shifting, rescaling, and flipping. Data augmentation enhances the model's ability to generalize and recognize malaria-infected cells in various orientations and conditions.

### Step 5:
We design a convolutional neural network (CNN) architecture consisting of convolutional layers, max-pooling layers, and fully connected layers. The model is compiled with appropriate loss and optimization functions.

### Step 6:
We split the dataset into training and testing sets, and then train the CNN model using the training data. The model learns to differentiate between parasitized and uninfected cells during this phase.

### Step 7:
We visualize the training and validation loss to monitor the model's learning progress and detect potential overfitting or underfitting.

### Step 8:
We evaluate the trained model's performance using the testing data, generating a classification report and confusion matrix to assess accuracy and potential misclassifications.

### Step 9:
We demonstrate the model's practical use by randomly selecting and testing a new cell image for classification.

## PROGRAM

```
Developed By: Pragatheesvaran AB
Register No: 212221240039
```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot





### Classification Report





### Confusion Matrix


### New Sample Data Prediction




## RESULT
The model's performance is evaluated through training and testing, and it shows potential for assisting healthcare professionals in diagnosing malaria more efficiently and accurately.
