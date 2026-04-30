# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset into a DataFrame and explore its contents to understand the data structure.

2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.

3.Create a linear regression model and fit it using the training data.

4.Predict the results for the testing set and plot the training and testing sets with fitted lines. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NIWASH K
RegisterNumber:  212225230205
*/
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv(r"C:\Users\acer\Downloads\student_scores.csv")
df.head(10)

plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
y

from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,Y_train)
x_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(x_train,lr.predict(x_train),color='red')
lr.coef_
lr.intercept_

y_pred=lr.predict(x_test)
mse=mean_squared_error(Y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print("MSE : ",mse)
print("RMSE : ",rmse)
print("MAE : ",mae)
print("R2 : ",r2)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="197" height="400" alt="17775112612632417303746759265242" src="https://github.com/user-attachments/assets/6091839a-3b1d-4692-8018-5a2f73f81004" />

<img width="661" height="807" alt="17775112911821908945323302573519" src="https://github.com/user-attachments/assets/ffe29b9b-def8-4bf2-b286-493cc6212281" />
<img width="632" height="433" alt="1777511319340493500596234375219" src="https://github.com/user-attachments/assets/052ad1f6-aab0-41d4-b08c-ad3f0f1dba2b" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
