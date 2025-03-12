# Titanic Survival Prediction Model

This project uses machine learning techniques to predict the survival of passengers aboard the Titanic based on various features like age, gender, class, family size, etc. It builds a model using **Linear Regression** to classify whether a passenger survived or not.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data Description](#data-description)
- [Model Explanation](#model-explanation)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

This repository demonstrates a **Titanic Survival Prediction Model** using a basic **Linear Regression** model to predict the survival outcome of passengers on the Titanic. The dataset used is the famous Titanic dataset available from Kaggle.

## Dependencies

The following libraries are required to run this project:

- `pandas` - For data manipulation and analysis
- `numpy` - For numerical operations
- `scikit-learn` - For machine learning and preprocessing
- `matplotlib` (optional, for visualization)


## Data Description

The data is from the **Titanic Dataset from kaggle** and contains several features about the passengers.

## Model Explanation

This project applies the following steps:

1. **Data Preprocessing**:
   - The dataset is loaded and basic information is displayed.
   - The `Sex` column is encoded into binary (1 for male, 0 for female).
   - A new feature, `FamSize`, is created by adding `SibSp` and `Parch` to represent the family size.
   - Irrelevant columns (`SibSp` and `Parch`) are dropped.

2. **Feature Selection**:
   - The target variable `Survived` is separated from the features, and the dataset is split into training and test sets.

3. **Data Scaling**:
   - The `StandardScaler` from `sklearn` is applied to scale features, which improves the performance of linear models.

4. **Linear Regression**:
   - A **Linear Regression** model is trained to predict survival based on the available features.

5. **Sigmoid Transformation**:
   - As we are predicting a binary outcome (survived or not), we apply the **sigmoid function** to the linear regression output to convert it into a probability.

6. **Model Evaluation**:
   - **Confusion Matrix** is calculated to evaluate the model performance on both training and test data.
   - Additional evaluation metrics, including **accuracy**, **precision**, **recall**, and **F1-score**, are computed for both training and test data.
  

## Results
After training the model, the script will output the confusion matrix and the following metrics for both training matrix and testing matrix. 

![Train Matrix1]
<img width="178" alt="image" src="https://github.com/user-attachments/assets/3fb6467e-2004-4452-a7ae-9a3afb6d8052" />

![Train Matrix2]
<img width="141" alt="image" src="https://github.com/user-attachments/assets/053732af-e992-4a0c-aee7-409ff69cf103" />


![Test Matrix1]
<img width="164" alt="image" src="https://github.com/user-attachments/assets/c8da6c97-cc41-40ec-be62-9f225aa5ee93" />

![Test Matrix1]
<img width="143" alt="image" src="https://github.com/user-attachments/assets/6af0f4e2-0139-4956-8228-3977d4d6cb10" />


## Conclusion

This project serves as a valuable learning experience for me, it helped me understand data preprocessing, feature engineering, model selection, the usecase of libraries which i was unfamiliar with, and how mathematics is connected with ML. I expect myself to improve my skills and build better and high accuracy models. 





## CODE BREAKDOWN

### 1. Importing and Libraries used 
``` python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```
 
### 2. Loading and Inspecting data
``` python
train_data = pd.read_csv('train.csv')

def desc_initial_Data():
    train_data.info()
    print(train_data.isnull().sum())
``` 

### 3. Data Preprocessing 
``` python
new_data['Sex'] = new_data['Sex'].map({'male': 1, 'female': 0})
new_data['FamSize'] = new_data['SibSp'] + new_data['Parch']
new_data = new_data.drop(columns=['SibSp', 'Parch'])
```

### 4. Splitting Data into Feature(X) and Target(Y)
``` python
.x = new_data.drop(columns=['Survived'])
y = new_data['Survived']
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
``` 


### 5.Scaling the data
``` python
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
x_test = scalar.transform(x_test)
```

### 6. Training the Model
``` python
ML_Model = LinearRegression().fit(X_train, Y_train)
z_train = ML_Model.predict(X_train)
z_test = ML_Model.predict(x_test)
``` 
### 7.Applying Sigmoid and Making Predictions
``` python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

surviving_prob_train = sigmoid(z_train)
surviving_prob_test = sigmoid(z_test)

manual_predictions_train = (surviving_prob_train >= 0.5).astype(int)
manual_predictions_test = (surviving_prob_test >= 0.5).astype(int)
``` 

### 8. Evaluating confusion matrix 
``` python
TP_train = sum((manual_predictions_train == 1) & (Y_train == 1)) 
TN_train = sum((manual_predictions_train == 0) & (Y_train == 0)) 
FP_train = sum((manual_predictions_train == 1) & (Y_train == 0))  
FN_train = sum((manual_predictions_train == 0) & (Y_train == 1))  




TP_test = sum((manual_predictions_test == 1) & (y_test == 1)) 
TN_test = sum((manual_predictions_test == 0) & (y_test == 0))  
FP_test = sum((manual_predictions_test == 1) & (y_test == 0))  
FN_test = sum((manual_predictions_test == 0) & (y_test == 1))

``` 

### 9. Printing accuracy, precision, recall, and f1 score. 
```python
print(f"Train Confusion Matrix:")
print(f"True Positives: {TP_train}")
print(f"True Negatives: {TN_train}")
print(f"False Positives: {FP_train}")
print(f"False Negatives: {FN_train}")

print(f"\nTest Confusion Matrix:")
print(f"True Positives: {TP_test}")
print(f"True Negatives: {TN_test}")
print(f"False Positives: {FP_test}")
print(f"False Negatives: {FN_test}")
print('\n')

accuracy_train = (TP_train + TN_train) / (TP_train + TN_train + FP_train + FN_train)
precision_train = TP_train / (TP_train + FP_train) if TP_train + FP_train != 0 else 0
recall_train = TP_train / (TP_train + FN_train) if TP_train + FN_train != 0 else 0
f1_score_train = 2 * (precision_train * recall_train) / (precision_train + recall_train) if precision_train + recall_train != 0 else 0

accuracy_test = (TP_test + TN_test) / (TP_test + TN_test + FP_test + FN_test)
precision_test = TP_test / (TP_test + FP_test) if TP_test + FP_test != 0 else 0
recall_test = TP_test / (TP_test + FN_test) if TP_test + FN_test != 0 else 0
f1_score_test = 2 * (precision_test * recall_test) / (precision_test + recall_test) if precision_test + recall_test != 0 else 0

print(f"Train Metrics:")
print(f"Accuracy: {accuracy_train:.4f}")
print(f"Precision: {precision_train:.4f}")
print(f"Recall: {recall_train:.4f}")
print(f"F1 Score: {f1_score_train:.4f}")

print(f"\nTest Metrics:")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_score_test:.4f}")
```

### Key Observations:
- **Accuracy and Precision** are low (around 0.39 for training and 0.41 for testing), suggesting that the model struggles to make correct predictions consistently.
- **Recall** is very high (close to 1), indicating that the model is good at identifying survivors but at the cost of precision, meaning it also makes many false positive predictions.
- The **F1 score** is moderately low (around 0.55 for training and 0.5850 for testing), highlighting an imbalance between precision and recall, which could be improved.

