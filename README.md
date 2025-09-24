# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
```.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("Employee.csv")
df=pd.DataFrame(data)
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
print(df.info())
print(df.isnull().sum())
data["left"].value_counts()
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours",
        "time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=df["left"]
y.head()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Name: Junjar U")
print("Register Number: 212224230110")
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
cm=metrics.confusion_matrix(y_test, y_pred)
cr=metrics.classification_report(y_test, y_pred)
print(accuracy)
print(cm)
print(cr)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])


```

## Output:
### Data head
<img width="1327" height="230" alt="image" src="https://github.com/user-attachments/assets/0901b544-9e69-4ca0-9236-a9e6b9401c18" />

### Dataset info
<img width="589" height="666" alt="image" src="https://github.com/user-attachments/assets/d6ce3e0b-12a3-486b-b34f-bac80eb54535" />

### Values count in left column
<img width="292" height="91" alt="image" src="https://github.com/user-attachments/assets/142bd84b-a7d8-4e24-b3ea-4e57e9d2d8bc" />

### Head value of x and y
<img width="848" height="448" alt="image" src="https://github.com/user-attachments/assets/be38dab1-5322-4f37-9443-aa8ba7529ad2" />

### Predicted data
<img width="484" height="99" alt="image" src="https://github.com/user-attachments/assets/99512079-c521-4e0b-ac2a-6076d3f832eb" />

### Accuracy
<img width="166" height="61" alt="image" src="https://github.com/user-attachments/assets/19a206e7-306a-41de-bff8-779ed70d8b35" />

### Confusion matrix
<img width="159" height="57" alt="image" src="https://github.com/user-attachments/assets/a17bd747-3eb3-4d5e-bf69-7a0882ffb375" />

### Classification report
<img width="615" height="210" alt="image" src="https://github.com/user-attachments/assets/10e892f2-74cd-4e2c-a7fa-2b3c49b72336" />

### Data prediction
<img width="1326" height="129" alt="image" src="https://github.com/user-attachments/assets/5fc7e96f-a3ca-459f-81e4-a8b517edc146" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
