# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

breast_cancer_dataset = load_breast_cancer()

df = pd.DataFrame(
    breast_cancer_dataset.data,
    columns=breast_cancer_dataset.feature_names
)

df['label'] = breast_cancer_dataset.target

print(df.head())

# ---------------------------------------------------
# SPLIT FEATURES AND LABELS
# ---------------------------------------------------

X = df.drop(columns='label', axis=1)
Y = df['label']

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

print(X.shape, X_train.shape, X_test.shape)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# LOGISTIC REGRESSION MODEL
# ---------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(
    Y_train,
    X_train_prediction
)

print('Accuracy on training data = ',
      training_data_accuracy)

X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(
    Y_test,
    X_test_prediction
)

print('Accuracy on test data = ',
      test_data_accuracy)

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

cm = confusion_matrix(Y_test, X_test_prediction)

print('\nConfusion Matrix:')
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.savefig('images/confusion_matrix.png')
plt.close()

# ---------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------

print('\nClassification Report:')

print(
    classification_report(
        Y_test,
        X_test_prediction
    )
)

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

feature_importance = feature_importance.sort_values(
    by='Coefficient',
    ascending=False
)

print('\nFeature Importance:')
print(feature_importance)

# ---------------------------------------------------
# FEATURE IMPORTANCE VISUALIZATION
# ---------------------------------------------------

top_features = feature_importance.head(10)

plt.figure(figsize=(10,8))

plt.barh(
    top_features['Feature'],
    top_features['Coefficient']
)

plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Top 10 Important Features')

plt.savefig('images/feature_importance.png')
plt.close()

# ---------------------------------------------------
# ROC CURVE + AUC SCORE
# ---------------------------------------------------

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    Y_test,
    y_prob
)

auc_score = roc_auc_score(
    Y_test,
    y_prob
)

print('\nAUC Score:', auc_score)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.savefig('images/roc_curve.png')
plt.close()

# ---------------------------------------------------
# RANDOM FOREST CLASSIFIER
# ---------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=2
)

rf_model.fit(X_train, Y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    Y_test,
    rf_predictions
)

print('\nRandom Forest Accuracy:',
      rf_accuracy)

print('\nRandom Forest Classification Report:')

print(
    classification_report(
        Y_test,
        rf_predictions
    )
)

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

pickle.dump(
    model,
    open('models/breast_cancer_model.pkl', 'wb')
)

print('\nModel saved successfully!')
