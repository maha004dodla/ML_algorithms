import pandas as pd

#data extraction
data=pd.read_csv(r"C:\Users\MAHA LAKSHMI\desktop tags\Downloads\crime.csv")


#converting into proper format day-month-yr hrs-min-seconds
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')    
data['timestamp'] = pd.to_datetime(data['timestamp'], format = '%d-%m/%Y %H:%M:%S')


#seperating timestamp as seperate columns
col_new = data.iloc[:,0]
db=pd.DataFrame({"year": col_new.dt.year,"month": col_new.dt.month,"day": col_new.dt.day,
                 "hour": col_new.dt.hour,"dayofyear":col_new.dt.dayofyear,
                 "week":col_new.dt.week,"weekofyear": col_new.dt.weekofyear,
                 "dayofweek": col_new.dt.dayofweek,"weekday": col_new.dt.weekday,
                 "quarter": col_new.dt.quarter,})


#removing timestamp columns after seperating into new columns
newdataset=data.drop('timestamp',axis=1)
data1=pd.concat([db,newdataset],axis=1)
data1=data1.dropna()


#seperating into independent and dependent variables
x=data1.iloc[:,[1,2,3,4,6,16,17]].values
y=data1.iloc[:,[10,11,12,13,14,15]].values


#splitting into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=9)


#model building    
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)


y_pred=model.predict(x_test)


#accuracy prediction
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred)*100)