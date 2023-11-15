# -Machine-learning-Cancer-cell-classification-using-Scikit-learn
## Features

- CNN and transfer learning applied for pneumonia detection.
- Real-time processing of medical images for timely diagnostics.
- Visual representation with overlays for enhanced interpretation.
- User-friendly interface for easy interaction with the detection system.
- Detailed diagnostic reports with probability scores and visual heatmaps.
## Requirements

- Python 3.x for project development.
- Essential Python packages: tensorflow, keras, opencv-python, numpy for image processing.

## Architecture Diagram



## Flow chart



## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/Paul-Andrew-15/Pneumonia-Detection-using-Convolutional-neural-network-and-Transfer-learning-Resnet50-.git

2. Install the required packages:

3. Download the pre-trained pneumonia detection model and label mappings.

## Usage

1. Open a new Google Colab notebook.

2. Upload the project files in Google Drive.

3. Load the pre-trained pneumonia detection model and label mappings. Ensure the model files are correctly placed in the Colab working directory.

4. Execute the Pneumonia Detection script in the Colab notebook, which may involve adapting the script to run within a notebook environment.

5. Follow the on-screen instructions or customize input cells in the notebook for pneumonia detection with uploaded medical images.

6. View and analyze the results directly within the Colab notebook.

7. Repeat the process for additional images or iterations as needed.

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
