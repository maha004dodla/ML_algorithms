import pandas as pd

#data extraction
data=pd.read_excel(r"C:\Users\MAHA LAKSHMI\desktop tags\Downloads\kmeans1.xlsx")
print(data.head())


#data understanding
print(data.shape)
print(data.columns)


#data preprocessing
data1=data.drop(['ID Tag','Model'],axis=1)
data1=data1.drop(['Department'],axis=1)
print(data1.shape)
print(data1.columns)
print(data1.head())


#model building
from sklearn.cluster import KMeans
km=KMeans(n_clusters=2,init='k-means++',n_init=10)
km.fit(data1)


#value prediction
x=km.fit_predict(data1)
print(x)


#assigning to new column
data['cluster']=x
data1=data.sort_values(['cluster'])
print(data1)


#converting it to new csv file
data1.to_csv('E:\KmeansPredicted.csv')
