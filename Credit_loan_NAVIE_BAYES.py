#importing packages
import pandas as pd
import numpy as np


#data extraction
indexes=[i for i in 'abcdefghijklmnop']
data=pd.read_csv(r"C:\Users\MAHA LAKSHMI\desktop tags\Downloads\credit_loan.csv",names=indexes)


#data understanding
print(data.head())
print(data.shape)
print(data.isna().sum())


#data preprocessing
#method: to find counts of '?' and replacing them

li=['?']
d1=data.isin(li).sum()
print(d1)


print(data['a'].value_counts())
data['a'].replace(to_replace='?',value='b',inplace=True)
print(data['a'].value_counts())


print(data['b'].value_counts())
data['b'].replace(to_replace='?',value=22.67,inplace=True)
print(data['b'].value_counts())


print(data['d'].value_counts())
data['d'].replace(to_replace='?',value='u',inplace=True)
print(data['d'].value_counts())


print(data['e'].value_counts())
data['e'].replace(to_replace='?',value='g',inplace=True)
print(data['e'].value_counts())

print(data['f'].value_counts())
data['f'].replace(to_replace='?',value='c',inplace=True)
print(data['f'].value_counts())


print(data['g'].value_counts())
data['g'].replace(to_replace='?',value='v',inplace=True)
print(data['g'].value_counts())


print(data['n'].value_counts())
data['n'].replace(to_replace='?',value=0,inplace=True)
print(data['n'].value_counts())


#converting string to integers
from sklearn.preprocessing import LabelEncoder
l_enc=LabelEncoder()
s='adefgijlm'
for i in s:
    data[i]=l_enc.fit_transform(data[i])
    
print(data.head())


#splitting to independent and target variables
x=np.array(data.iloc[:,:-1])
y=data.iloc[:,-1].values
print(x.shape,y.shape)

#removing outliers
#method 1: normalization
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
scaledx=scale.fit_transform(x)


#method2: standardization
#from sklearn.preprocessing import StandardScaler
#ss=StandardScaler()
#scaledx=ss.fit_transform(x)



#model building
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(scaledx,y,test_size=0.2)

print(xtrain.shape,xtest.shape)
print(ytrain.shape,ytest.shape)



from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)


from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)



