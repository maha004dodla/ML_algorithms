import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data extraction
indexes=[i for i in 'abcdefghijklmn']
data=pd.read_csv(r"C:\Users\MAHA LAKSHMI\desktop tags\Downloads\boston.csv",names=indexes)

#data understanding
print(data.head())
print(data.columns)
print(data.shape)


#data prepossing
print(data.isna().sum())
print(data.describe())
data.dropna(axis=0,inplace=True)
print(data.isna().sum())
print(data.shape)

a=data.corr()
plt.matshow(a)
plt.colorbar()
plt.show()


#converting an dataframe into array by using array() and values method
x=np.array(data.iloc[ : , :-1])
y=data.iloc[ : ,-1].values
print(x)
print(y)
print(x.shape)
print(y.shape)


#splitting the data into testing and training
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=30)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import mean_squared_error
import math
print(math.sqrt(mean_squared_error(ytest,ypred)))

