# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Read the given dataset and assign x and y array.
3. Split x and y into training and test set.
4. Split x and y into training and test set.
5. Fit the logistic regression for the training set to predict y.
6. Create the confusion matrix and find the accuracy score, recall sensitivity and specificity.
7. Plot the training set results.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: D.Amarnathreddy
RegisterNumber: 212221240012 


#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading and displaying dataframe
df=pd.read_csv("Social_Network_Ads (1).csv")
df

#assigning x and y and displaying them
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values 

#splitting data into train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

#scaling values and obtaining scaled array of train and test of x
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)

#applying logistic regression to the scaled array
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)

#finding predicted values of y
ypred=c.predict(xtest)
ypred

#calculating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm

#calculating accuracy score
from sklearn import metrics
acc=metrics.accuracy_score(ytest,ypred)
acc

#calculating recall sensitivity and specificity
r_sens=metrics.recall_score(ytest,ypred,pos_label=1)
r_spec=metrics.recall_score(ytest,ypred,pos_label=0)
r_sens,r_spec

#displaying regression 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
xs,ys=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xs[:,0].min()-1,stop=xs[:,0].max()+1,step=0.01),
               np.arange(start=xs[:,1].min()-1,stop=xs[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,c.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,cmap=ListedColormap(("pink","purple")))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x1.max())
for i,j in enumerate(np.unique(ys)):
    plt.scatter(xs[ys==j,0],xs[ys==j,1],
                c=ListedColormap(("white","violet"))(i),label=j)
plt.title("Logistic Regression(Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
*/
```

## Output:
<img width="485" alt="1" src="https://user-images.githubusercontent.com/94165103/173886528-729f640c-59d9-4b50-8f5f-38c89b0b84d4.png">

<img width="623" alt="2" src="https://user-images.githubusercontent.com/94165103/173886584-15883833-7e58-4b46-a88e-d8d220bed10f.png">

<img width="443" alt="3" src="https://user-images.githubusercontent.com/94165103/173886622-39c15542-9e84-43d0-afdc-1a1687cb8f89.png">

<img width="311" alt="4" src="https://user-images.githubusercontent.com/94165103/173886664-fa04c961-a959-479f-931e-adf1ee7ee24b.png">

<img width="442" alt="5" src="https://user-images.githubusercontent.com/94165103/173886704-441beb96-31c6-454f-8563-0e0aa46a5fa2.png">

<img width="562" alt="6" src="https://user-images.githubusercontent.com/94165103/173886755-636f0489-146c-4ddb-9e8d-161e67843c9e.png">


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

