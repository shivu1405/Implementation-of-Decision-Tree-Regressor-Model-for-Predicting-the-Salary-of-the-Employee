# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset containing employee details like experience, test score, and interview score.

2. Split the dataset into input features (X) and target variable (Salary), then divide it into training and testing sets.

3. Create and train a Decision Tree Regressor model using the training data.

4. Predict the salary for test data and evaluate the model’s performance using metrics like Mean Squared Error and R² Score.
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Shivasri
RegisterNumber:  212224220098
*/
```
```
# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a sample dataset
data = {
    'Experience': [1, 3, 5, 7, 9, 11, 13, 15],
    'Test_Score': [80, 86, 82, 90, 87, 95, 98, 99],
    'Interview_Score': [75, 78, 82, 85, 86, 88, 92, 94],
    'Salary': [40000, 50000, 60000, 80000, 85000, 90000, 95000, 100000]
}

df = pd.DataFrame(data)

# Step 2: Separate the features (X) and target (y)
X = df[['Experience', 'Test_Score', 'Interview_Score']]
y = df['Salary']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 4: Create the Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=1)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict the salary
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 8: Display results
print("Actual Salaries:", list(y_test))
print("Predicted Salaries:", list(y_pred))
print("\nMean Squared Error:", mse)
print("R² Score:", r2)

```

## Output:
<img width="1036" height="648" alt="image" src="https://github.com/user-attachments/assets/324f30db-2cd0-4860-ad1c-cdfb92006627" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
