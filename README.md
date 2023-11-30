# CANCER CELL DETECTION USING MACHINE LEARNING

Machine Learning is a sub-field of Artificial Intelligence that gives systems the ability to learn themselves without being explicitly programmed to do so. Machine Learning can be used in solving many real world problems. 
Let’s classify cancer cells based on their features, and identifying them if they are ‘malignant’ or ‘benign’. We will be using scikit-learn for a machine learning problem. Scikit-learn is an open-source machine learning, data mining and data analysis library for Python programming language.
## Features

- Machine learning models can analyze large datasets and identify patterns that may not be apparent to human observers. This enables early detection of cancer cells, potentially leading to more successful treatment outcomes.
- Automated classification using machine learning algorithms can process and analyze large volumes of medical data much faster than manual methods. This efficiency is crucial for timely decision-making in a clinical setting.
- Machine learning models can be continuously trained and updated as more data becomes available. This adaptability allows the model to improve over time, staying relevant as new information and technologies emerge.
- User-friendly interface for easy interaction with the detection system.
- The analysis of large datasets using machine learning can reveal novel insights into cancer biology and behavior. This can drive further research and contribute to a deeper understanding of the disease
## Requirements

- Python 3.x for project development.

## Flow chart
![image](https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/95266924/88cd72ea-9bda-4121-9495-745605b09f6a)



## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn.git

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

### label names:
![WhatsApp Image 2023-11-15 at 21 14 35 (3)](https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/95266924/d15c3db9-f1e2-4d00-b98d-0e239861b179)

### Labels:
![WhatsApp Image 2023-11-15 at 21 14 35 (2)](https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/95266924/5f2dc05b-b826-46ef-9b9a-356101035728)

### Feature names:
![WhatsApp Image 2023-11-15 at 21 14 35 (1)](https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/95266924/7a000d24-786e-4370-b670-79ab76cd80a7)

### Features:
![WhatsApp Image 2023-11-15 at 21 14 35](https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/95266924/f09bb376-1cae-489e-aaa3-5bb5d9d22ce1)

### Features:
![image](https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/95266924/b1031495-c445-4bdc-ad89-c32f52b8f1da)

### Prediction:
![image](https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/95266924/e64d7173-5a83-443d-98c7-03ef317b582a)

### Accuracy:
![WhatsApp Image 2023-11-15 at 21 14 35 (4)](https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/95266924/b1e99417-32f0-4ede-be37-7f36ba52a043)

### Plot - 1:
<img width="416" alt="8" src="https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/104237399/188739e6-629a-487d-9224-ec163f2ec0c4">

### Plot - 2:
<img width="424" alt="9" src="https://github.com/praga-16/-Machine-learning-Cancer-cell-classification-using-Scikit-learn/assets/104237399/5fdb2e98-081f-45d2-814b-d54f60d1ac4b">

## Result:
The ML | Cancer cell classification using Scikit-learn,demonstrates strong performance on both training and testing datasets:

- The model achieved an accuracy of 96.15% on the test dataset, showcasing its ability to correctly classify cases.
- This machine learning classifier based on the Naive Bayes algorithm is 94.15% accurate in predicting whether a tumor is malignant or benign.

These results suggest that the cancer cell model is both accurate and well-balanced, with high precision and recall values.
