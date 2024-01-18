#importing packages
import pandas as pd
import numpy as np


#data extraction
data=pd.read_csv(r"C:\Users\MAHA LAKSHMI\desktop tags\Downloads\MetroPT3(AirCompressor).csv")


#data understanding
print(data.head())
print(data.shape)
print(data.isna().sum())
print(data.columns)


#data preprocessing
li=['?']
d1=data.isin(li).sum()
print(d1)

data.drop('timestamp',axis=1,inplace=True)
print(data.shape)


#splitting into x and y values
x=np.array(data.iloc[:,:-1])
y=data.iloc[:,-1].values
print(x.shape,y.shape)


#model building
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

print(xtrain.shape,xtest.shape)
print(ytrain.shape,ytest.shape)


#model training
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy')
model.fit(xtrain,ytrain)

#predicting values
ypred=model.predict(xtest)


#accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)


#model saving
import pickle
sample=r'E:\python programs\saved_models_ML\air_compressor_model.sav'
pickle.dump(model,open(sample,'wb'))


#model importing 
import pickle
model_sample=pickle.load(open(r'E:\python programs\saved_models_ML\air_compressor_model.sav','rb'))
pred=model_sample.predict([[19250,-0.014,9.362,9.35,-0.024,9.366,53.325,0.04,1,0,1,1,0,1,1]])
print(pred)









