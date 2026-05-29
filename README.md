# Breast Cancer Classification using Machine Learning

## Overview

This project implements a Breast Cancer Classification model using Logistic Regression in Python. The model predicts whether a breast tumor is:

* Malignant (Cancerous)
* Benign (Non-cancerous)

The dataset used is the Breast Cancer Wisconsin Diagnostic Dataset available in scikit-learn.

---

# Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

# Machine Learning Workflow

1. Load Breast Cancer Dataset
2. Convert dataset into Pandas DataFrame
3. Perform Exploratory Data Analysis (EDA)
4. Check missing values and statistical measures
5. Split dataset into training and testing data
6. Apply feature scaling using StandardScaler
7. Train Logistic Regression model
8. Evaluate model performance
9. Generate confusion matrix and classification report
10. Predict tumor type for new input data

---

# Dataset Information

| Property      | Value              |
| ------------- | ------------------ |
| Total Samples | 569                |
| Features      | 30                 |
| Classes       | Malignant / Benign |

---

# Data Distribution

| Label         | Count |
| ------------- | ----- |
| Benign (1)    | 357   |
| Malignant (0) | 212   |

---

# Model Performance

## Training Accuracy

```python
Accuracy on training data = 0.989
```

## Testing Accuracy

```python
Accuracy on test data = 0.973
```

---

# Confusion Matrix

Example confusion matrix:

```text
[[40  2]
 [ 1 71]]
```

Interpretation:

* 40 malignant tumors correctly predicted
* 71 benign tumors correctly predicted
* 3 incorrect predictions

---

# Classification Report

```text
              precision    recall  f1-score   support

           0       0.98      0.95      0.96        42
           1       0.97      0.99      0.98        72

    accuracy                           0.97       114
   macro avg       0.98      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

---

# Example Prediction

```text
The Breast Cancer is Benign
```

OR

```text
The Breast cancer is Malignant
```

---

# Important Code Improvement

The input prediction should also be scaled before prediction.

Correct code:

```python
input_data_scaled = scaler.transform(input_data_reshaped)
prediction = model.predict(input_data_scaled)
```

---

# Project Structure

```text
Breast-Cancer-Classification/
│
├── images/
│   └── confusion_matrix.png
│
├── notebooks/
│   └── breast_cancer_analysis.ipynb
│
├── src/
│   └── breast_cancer_classifier.py
│
├── requirements.txt
├── README.md
└── results.txt
```

---

# How to Run the Project

## Clone Repository

```bash
git clone <repository_link>
cd Breast-Cancer-Classification
```

## Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Model

```bash
python src/breast_cancer_classifier.py
```

---

# Future Improvements

* ROC-AUC Curve
* Cross Validation
* Random Forest Classifier
* Support Vector Machine (SVM)
* Deep Learning Models
* Feature Importance Visualization
* Streamlit Web Application

---

# Conclusion

This project demonstrates a complete machine learning pipeline for breast cancer classification using Logistic Regression. The model achieves high accuracy and provides a strong foundation for medical machine learning and healthcare AI applications.
