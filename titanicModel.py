import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train.csv')

def desc_initial_Data():
    train_data.info()
    print(train_data.isnull().sum())


new_data = pd.DataFrame(train_data,columns=['Pclass','Sex','SibSp','Parch','Survived'])

def desc_New_Data():
    new_data.info()
    print(new_data.isnull().sum())



#filtering data 
new_data['Sex'] =new_data['Sex'].map({'male':1,'female':0})
new_data['FamSize'] = new_data['SibSp'] + new_data['Parch']
new_data = new_data.drop(columns=['SibSp','Parch'])
   
#for x we get y 
x = new_data.drop(columns=['Survived'])
y = new_data['Survived']

X_train,x_test,Y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#standard scaling 
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
x_test = scalar.transform(x_test)

#Linear Regression, ML Model 
#i don't understand how binary data and continious data are used to produce the result 
#this model most probably is not accurate 
#learns features and relationshp between features and target values 
ML_Model = LinearRegression().fit(X_train,Y_train)

#z = x * coeff + intercept 
#the model has already been trained it knows the relationship beween feature and target values
#therefore it can make predictions for x train and x test
z_train = ML_Model.predict(X_train)
z_test = ML_Model.predict(x_test)


#using sigmoid instead of library 
def sigmoid(z):
    return 1/(1+np.exp(-z))

#probabilities from sigmoid

surviving_prob_train = sigmoid(z_train)
surviving_prob_test = sigmoid(z_test)

manual_predictions_train = (surviving_prob_train >= 0.5).astype(int)
manual_predictions_test = (surviving_prob_test >= 0.5).astype(int)

TP_train = sum((manual_predictions_train == 1) & (Y_train == 1)) 
TN_train = sum((manual_predictions_train == 0) & (Y_train == 0)) 
FP_train = sum((manual_predictions_train == 1) & (Y_train == 0))  
FN_train = sum((manual_predictions_train == 0) & (Y_train == 1))  




TP_test = sum((manual_predictions_test == 1) & (y_test == 1)) 
TN_test = sum((manual_predictions_test == 0) & (y_test == 0))  
FP_test = sum((manual_predictions_test == 1) & (y_test == 0))  
FN_test = sum((manual_predictions_test == 0) & (y_test == 1))


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

# Calculate metrics for Train
accuracy_train = (TP_train + TN_train) / (TP_train + TN_train + FP_train + FN_train)
precision_train = TP_train / (TP_train + FP_train) if TP_train + FP_train != 0 else 0
recall_train = TP_train / (TP_train + FN_train) if TP_train + FN_train != 0 else 0
f1_score_train = 2 * (precision_train * recall_train) / (precision_train + recall_train) if precision_train + recall_train != 0 else 0

# Calculate metrics for Test
accuracy_test = (TP_test + TN_test) / (TP_test + TN_test + FP_test + FN_test)
precision_test = TP_test / (TP_test + FP_test) if TP_test + FP_test != 0 else 0
recall_test = TP_test / (TP_test + FN_test) if TP_test + FN_test != 0 else 0
f1_score_test = 2 * (precision_test * recall_test) / (precision_test + recall_test) if precision_test + recall_test != 0 else 0

# Print metrics for train and test
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