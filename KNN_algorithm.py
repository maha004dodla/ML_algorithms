import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#data extraction
data=pd.read_csv(r'C:\Users\MAHA LAKSHMI\desktop tags\Downloads\iris.csv')


#data understanding
print(data.head())
print(data.columns)
print(data.shape)


#data prepossing
print(data.isna().sum())
print(data.describe())

#replacing null values with mean
data.fillna(data.mean(),inplace=True)
print(data.isna().sum())

#replacing null values with 0
##data.fillna(0,inplace=True)
##print(data.isna().sum())

#dropping the rows which are having null values
##data.dropna(axis=0,inplace=True)
##print(data)

#creating a new column with sequence numbers
data['ranges'] = np.arange(1,151)
print(data.shape)
print(data.head())

#data correlation
a=data.corr()
plt.matshow(a)
plt.colorbar()
plt.show()


#converting an dataframe into array by using array() and values method
#if additional column is not added 
x=np.array(data.iloc[ : , :-1])
y=data.iloc[ : ,-1].values
print(x)
print(y)
print(x.shape)
print(y.shape)

#if additional column is added
x=np.array(data.iloc[ : ,[0,1,2,3,5]])
y=data.iloc[ : ,-2].values
print(x)
print(y)
print(x.shape)
print(y.shape)




#splitting the data into testing and training
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

#if we mention random_state then same set of values will be going to testing and training all the time
#print('xtrain values')
#print(xtrain)
#print(xtrain.shape)

#print('ytrain')
#print(ytrain)
#print(ytrain.shape)

#print('xtest')
#print(xtest)
print(xtest.shape)

#print('ytest')
#print(ytest)
print(ytest.shape)


#model building
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3) #here k=3
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

new_value=model.predict([[1.4,6.9,4.6,8.8,118],[1,2,3,4,49],[0,0,0,0,100],[5,6,7,8,19]])
print(new_value)




