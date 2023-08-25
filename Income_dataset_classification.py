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
#replacing '?' with its highest repeating values
print(data['workclass'].value_counts())
data['workclass'].replace(to_replace=' ?',value=' Private',inplace=True)
print(data['workclass'].value_counts())


print(data['occ'].value_counts())
data['occ'].replace(to_replace=' ?',value=' Exec-managerial',inplace=True)
print(data['occ'].value_counts())


print(data['n_cty'].value_counts())
data['n_cty'].replace(to_replace=' ?',value=' United-States',inplace=True)
print(data['n_cty'].value_counts())



data['x']=np.arange(0,len(data))

#changing all columns from string to integers
#method 1:label encoding

from sklearn.preprocessing import LabelEncoder
l_enc=LabelEncoder()
s=['workclass','edu','mrg_st','occ','rel','race','sex','n_cty']
for i in s:
    data[i+'_enc']=l_enc.fit_transform(data[i])


s=data.drop_duplicates('workclass').set_index('x')
print(s[['workclass','workclass_enc']])


s1=data.drop_duplicates('edu').set_index('x')
print(s1[['edu','edu_enc']])


s2=data.drop_duplicates('mrg_st').set_index('x')
print(s2[['mrg_st','mrg_st_enc']])


s3=data.drop_duplicates('occ').set_index('x')
print(s3[['occ','occ_enc']])


s4=data.drop_duplicates('rel').set_index('x')
print(s4[['rel','rel_enc']])


s5=data.drop_duplicates('race').set_index('x')
print(s5[['race','race_enc']])


s6=data.drop_duplicates('sex').set_index('x')
print(s6[['sex','sex_enc']])


s7=data.drop_duplicates('n_cty').set_index('x')
print(s7[['n_cty','n_cty_enc']])


#method 2: one hot encoding
#from sklearn.preprocessing import OneHotEncoder
#o_hot=OneHotEncoder()
#df=pd.DataFrame(o_hot.fit_transform(data[['workclass']]).toarray())
#data=data.join(df)
#print(data)


#droppping string labled columns
data.drop(['workclass', 'edu', 'mrg_st', 'occ', 'rel', 'race', 'sex', 'n_cty'],axis=1,inplace=True)


#data splitting into training and testing
x=np.array(data.iloc[:,[0,1,2,3,4,5,8,9,10,11,12,13,14,15]]) 
y=data.iloc[:,6].values


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
model=KNeighborsClassifier(n_neighbors=15)
model.fit(xtrain,ytrain)


#predicting values
ypred=model.predict(xtest)


#accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)


#predicting new values
#print(data.head())
#give the values in which algorithm is trained
#order=['age', 'fnlwgt', 'edu_num', 'c_gain', 'c_loss', 'HPW', 'income', 'x','workclass_enc', 'edu_enc', 'mrg_st_enc', 'occ_enc', 'rel_enc','race_enc', 'sex_enc', 'n_cty_enc']
a=[50,83311,13,0,0,13,5,9,2,3,0,4,1,38]
new_value=model.predict([a])
print(new_value)



#execu           
#0.2 10 80.3  
#0.2 15 80.5
#0.3 10 80.5
#0.3 15 80.7


#craf
#0.2 15 80.5
#0.3 15 80.7
#0.2 10 80.3
#0.3 10 80.5


#Prof
#0.2 10 80.3
#0.3 10 80.5
#0.2 15 80.5
#0.3 15 80.7
