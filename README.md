# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
import dataset and get data info check for null values Map values for position column Split the dataset into train and test set Import decision tree regressor and fit it for data Calculate MSE,R2 and y predict.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SANGAVI SURESH
RegisterNumber: 212222230130 
*/

import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l0=LabelEncoder()

data["Position"]=l0.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

# DATA.HEAD():
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541861/b6bfb429-255d-4ea5-b08c-61018ca7d580)

# DATA.INFO():
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541861/8b9a3db2-f9f1-4354-be12-2f41e2f1bcac)

# ISNULL() AND SUM():
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541861/79936e21-a687-4288-abf6-9f35799c492d)

# DATA.HEAD() FOR SALARY:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541861/fa6e0d33-0fa9-4b73-8db4-e0cb245970f2)

# MSE VALUE:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541861/bbd99ebb-0a18-4ecb-a5d7-db879a9757fd)

# R2 VALUE:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541861/a7707446-38d2-4fba-9ccb-12200ae97ad7)

# DATA PREDICTION:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541861/1c1a3a69-b80a-443c-9024-1b3d301f3cf5)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
