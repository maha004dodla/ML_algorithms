import pandas as pd
import numpy as np

#data extraction
data=pd.read_csv(r"C:\Users\MAHA LAKSHMI\desktop tags\Downloads\survey lung cancer.csv")


#data preprocessing
print(data.head())
print(data.shape)
print(data.columns)
print(data.isna().sum())

#for changing string to numbers bcauz algo dont accept str datatype
#method 1 : data['GENDER']=data['GENDER'].map({'M':0,'F':1})
#method 2: label encoding

from sklearn.preprocessing import LabelEncoder
l_enc=LabelEncoder()
data['GENDER']=l_enc.fit_transform(data['GENDER'])
#print(data['GENDER'])


#data splitting into training and testing
x=np.array(data.iloc[ : , :-1])
y=data.iloc[:,-1].values

print(x.shape)
print(y.shape)


#model building
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.4)


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3) #here k=3
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)


#accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)


#result prediction
new_value=model.predict([[2,89,2,1,2,1,2,1,2,2,1,1,1,1,2]])
print(new_value)