# -Machine-learning-Cancer-cell-classification-using-Scikit-learn

Machine Learning is a sub-field of Artificial Intelligence that gives systems the ability to learn themselves without being explicitly programmed to do so. Machine Learning can be used in solving many real world problems. 
Let’s classify cancer cells based on their features, and identifying them if they are ‘malignant’ or ‘benign’. We will be using scikit-learn for a machine learning problem. Scikit-learn is an open-source machine learning, data mining and data analysis library for Python programming language.
## Features

- CNN and transfer learning applied for pneumonia detection.
- Real-time processing of medical images for timely diagnostics.
- Visual representation with overlays for enhanced interpretation.
- User-friendly interface for easy interaction with the detection system.
- Detailed diagnostic reports with probability scores and visual heatmaps.
## Requirements

- Python 3.x for project development.

## Flow chart
![image](https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/95266924/88cd72ea-9bda-4121-9495-745605b09f6a)



## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/Paul-Andrew-15/Pneumonia-Detection-using-Convolutional-neural-network-and-Transfer-learning-Resnet50-.git

2. For this machine learning project, we will be needing the ‘Scikit-learn’ Python module. If you don’t have it installed on your machine, download and install it by running the following
 pip install scikit-learn
3. Use any IDE for this project, by it is highly recommended Jupyter notebook for the project.
 pip install jupyter

## Usage

1. Open a new Google Colab notebook.

2. Importing the necessary module and dataset.

3. Loading the dataset to a variable. 

4. Organizing the data and looking at it. 

5. Organizing the data into Sets.

6. import the GaussianNB module and initialize it using the GaussianNB() function.

7.Evaluating the trained model’s accuracy.

## Program:

```python
# importing the Python module
import sklearn

# importing the dataset
from sklearn.datasets import load_breast_cancer

# loading the dataset
data = load_breast_cancer()

# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# looking at the data
print(label_names)

print(labels)

print(feature_names)

print(features)

# importing the function
from sklearn.model_selection import train_test_split

# splitting the data
train, test, train_labels, test_labels = train_test_split(features, labels,
									test_size = 0.33, random_state = 42)

# importing the module of the machine learning model
from sklearn.naive_bayes import GaussianNB

# initializing the classifier
gnb = GaussianNB()

# training the classifier
model = gnb.fit(train, train_labels)

# making the predictions
predictions = gnb.predict(test)

# printing the predictions
print(predictions)

# importing the accuracy measuring function
from sklearn.metrics import accuracy_score

# evaluating the accuracy
print(accuracy_score(test_labels, predictions))



``` 
## Output:

### Training log:

### Model evaluation metrics:


### Confusion matrix:


### Classification report:


## Result:

The pneumonia detection model, utilizing CNN and transfer learning with ResNet50, demonstrates strong performance on both training and testing datasets:

- The model achieved an accuracy of 89.26% on the test dataset, showcasing its ability to correctly classify pneumonia cases.
- During training, the model reached a high accuracy of 96.05% on the training dataset, indicating effective learning and generalization.
- Precision, measuring the model's ability to correctly identify positive cases, is notably high at 96.79%.

These results suggest that the pneumonia detection model is both accurate and well-balanced, with high precision and recall values. Further analysis, including the examination of the confusion matrix and visualizations provide additional insights into the model's performance.
