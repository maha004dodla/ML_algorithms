import pandas as pd
import numpy as np
#data extraction
data=pd.read_csv(r"C:\Users\MAHA LAKSHMI\desktop tags\Downloads\fires.csv")


#data understanding
print(data.head())
print(data.columns)
print(data.shape)


#data preprocessing
print(data.isna().sum())
data.dropna(axis=0,inplace=True)
print(data.isna().sum())
print(data.shape)


#data splitting into training and testing
x=np.array(data.iloc[:,:-1])
y=data.iloc[:,-1].values
print(x.shape)
print(y.shape)
print(x)
print(y)


#model building
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5) #if k is high accuracy is low
model.fit(xtrain,ytrain)


#predicting values for testing 
ypred=model.predict(xtest)


#calculating accuracy for our input and ouputs
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)


#result prediction
new_value=model.predict([[23,4,2013,28.9,69,7.8,9.4,6.8,15,15.8,2,4.6,0.9]])
print(new_value)




