#importing packages
import pandas as pd
import numpy as np

#data extraction
#adding columns names to the dataset
indexes=['age','workclass','fnlwgt','edu','edu_num','mrg_st','occ','rel','race','sex','c_gain','c_loss','HPW','n_cty','income']
data=pd.read_csv(r"C:\Users\MAHA LAKSHMI\desktop tags\Downloads\income.csv",names=indexes)


#data understanding
print(data.shape)
print(data.head())
print(data.columns)


#data preprocessing
#changing all columns from string to integers
from sklearn.preprocessing import LabelEncoder
l_enc=LabelEncoder()
print(data.isna().sum())
for i in data.columns:
    data[i]=l_enc.fit_transform(data[i])


#data splitting into training and testing
x=np.array(data.iloc[:,:-1])
y=data.iloc[:,-1].values

#printing independent and target size
print(x.shape)
print(y.shape)
print(x)
print(y)


#model building
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)

#training the algorithm
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=10)
model.fit(xtrain,ytrain)


#predicting values
ypred=model.predict(xtest)


#accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

#predicting new values
#print(data.head())
new_value=model.predict([[65,3,72471,4,16,3,9,2,2,1,82354,0,56,15],[65,7,5,11,2,5,5,2,41,23,22,6,4,20]])
print(new_value)








































