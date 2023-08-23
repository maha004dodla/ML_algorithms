from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 1a1)Load the .csv file into the notebook.
data1 = pd.read_csv(r'C:\Users\MAHA LAKSHMI\desktop tags\Downloads\public-covid-19-cases-canada.csv')
print(data1.head())



# 1a2)Summarize the information that can be derived from the dataset, including key features (columns), range of values, and useless values. All the information should be derived by necessary Python codes.
data1 = pd.read_csv(r'C:\Users\MAHA LAKSHMI\desktop tags\Downloads\public-covid-19-cases-canada.csv')
#printing and summarizing details of the csv file
#columns in the csv file are
print('columns in the csv file are:\n', data1.columns)
#size of the csv file is
print('size of the csv file is: \n', data1.index)
#empty values in each columns
print('empty values in each columns:\n', data1.isna().sum())
#dimensions of csv (rows,cols) 
print('dimensions of csv (rows,cols) :\n', data1.shape)
#printing particular column: 
print('printing particular column: \n', data1['age'].describe)
#first 10 lines are
print('first 10 lines are:\n', data1.head(10))
#last five lines
print('last 5 lines are:\n', data1.tail())



# 1a3)Explain TWO (2) potential insights that can be derived from the dataset.
# replacing null,not reported values
print('dataset after replacing the Not_reported values in gender column:\n')
data1['sex'].replace(to_replace='Not Reported',value=data1['sex'].mode()[0], inplace=True)

# insight: age dp to cases
data1 = data1[~data1['age'].isin(['NULL', 'Not Reported', 'NaN'])]
# plotting a bar graph for the age and distributions columns
data1['age'].value_counts().plot(kind='bar', xlabel='Age',ylabel='cases Count', title='Age Distribution')
plt.show()

# plotting graph for gender proportion
data1 = data1[~data1['sex'].isin(['NULL', 'Not Reported', 'NaN'])]
data1['sex'].value_counts().plot(kind='pie', title='gender proportion')
plt.show()



# 1b1)remove all the useless item like ‘NULL’, ‘Not Report’, etc.
data1 = pd.read_csv(r'C:\Users\MAHA LAKSHMI\desktop tags\Downloads\public-covid-19-cases-canada.csv')
print('before replacing case id:\n', data1)
print(data1.shape)
# filling all empty columns with sequence numbers using narange statement
print('Filling all empty case_id columns:\n')
a = np.arange(1, 50983)
data1['case_id'] = a
print(data1.shape)
print('empty values before removing columns:\n', data1.isna().sum())

# replacing null values with 'None' value
data1['has_travel_history'].fillna('None', inplace=True)
data1['locally_acquired'].fillna('None', inplace=True)
print('after replacing empty values count is:\n', data1.isna().sum())



# 1b2)Reformat the Age group as {‘0-19’, ’20-29’, ’30-39’, …, ’90-99’} and update the ‘age’ column
print('before reformatting data is:\n', data1['age'])
print('before removing useless values in age column dimensions are: ',data1['age'].shape)
# '~' is used to not consider the given input values like null,not reported
data1 = data1[~data1['age'].isin(['NULL', 'Not Reported', 'NaN'])]
print('after removing useless values in age column dimensions are: ',
      data1['age'].shape)

print('total age columns before reformatting: ', data1['age'].shape)

# creating a function a replace the input values with its corresponding ranges
def reformat(n):
    ins = {1: '0-19', 2: '20-29', 3: '30-39', 4: '40-49',
           5: '50-59', 6: '60-69', 7: '70-79', 8: '80-89', 9: '90-99'}
    if(n < 20):
        return ins[1]
    elif(n >= 20 and n < 30):
        return ins[2]
    elif(n >= 30 and n < 40):
        return ins[3]
    elif(n >= 40 and n < 50):
        return ins[4]
    elif(n >= 50 and n < 60):
        return ins[5]
    elif(n >= 60 and n < 70):
        return ins[6]
    elif(n >= 70 and n < 80):
        return ins[7]
    elif(n >= 80 and n < 90):
        return ins[8]
    else:
        return ins[9]

# assigning age column to age variable
age = data1['age']
for i in age:
    if '<' in i:
        a = i[1:]
        b = reformat(int(a))
        age.replace(to_replace=i, value=str(b), inplace=True)
    if(len(i) == 1 or len(i) == 2):
        c = reformat(int(a))
        age.replace(to_replace=i, value=str(c), inplace=True)
    if (i == '10-19'):
        d = reformat(10)
        age.replace(to_replace=i, value=str(d), inplace=True)

print(data1['age'])
print('total age columns after reformatting: ', data1['age'].shape)



# 1b3)List out the total number of infected person for each age group.
# method 1: value_count() method  to evaluate count of particular range distributions
age_counts = data1['age'].value_counts()
print('total number of infected person for each age group:\n', age_counts)
# method 2:
# inf=data1.groupby('age').count()
# print(inf['provincial_case_id'])



# (1c) Save the cleaned and reformatted dataset into a new .csv file.
print('creating a new csv above reformatting all tha changes in the input file')
# creating a new csv file
data1.to_csv('new_data.csv', header=True, index=False)
new = pd.read_csv('new_data.csv', index_col=0)
# printing its relations like columns,shape etc
print(new['age'].value_counts())
print(new.columns)
print(new.shape)
# creating a csv file in particular location
print('creating a new csv above reformatting all tha changes in the input file in specific location:\n')
data1.to_csv(r'E:\new_public_covid_file.csv')





# question2
# 2a)Design and apply a Python ORM(s) (Object Relation Mapping) to store the .csv file obtained in Q1(c). Please specify a table class before inserting the values into the database.
# object class name
Base = declarative_base()
class CovidCase(Base):
    __tablename__ = 'covid_cases'
    case_id = Column(Integer, primary_key=True)
    provincial_case_id = Column(Integer)
    age = Column(String)
    sex = Column(String)
    health_region = Column(String)
    province = Column(String)
    country = Column(String)
    date_report = Column(String)
    report_week = Column(String)
    has_travel_history = Column(String)
    locally_acquired = Column(String)
    case_source = Column(String)

# connecting to sqlite database
table1 = create_engine('sqlite:///covid_data.db')

# Create tables
Base.metadata.create_all(table1)

# Reading the csv file as dataframe
df = pd.read_csv(r'E:\new_public_covid_file.csv', index_col=0)

# Creating a session to interact with database
Session = sessionmaker(bind=table1)
session = Session()

# Iterate through DataFrame rows and insert data into the database
df.to_sql('covid_cases', con=table1, if_exists='append', index=False)

# Commit the changes and close the session
session.commit()
session.close()



# 2b)Compose queries on the database and answer the following questions:
# 2bi)What is the total number of male and female infectors for each month?
qdata = pd.read_csv('new_data.csv', index_col=0)
#creating a dataframe
df = pd.DataFrame(qdata)
# print(df)
#creating a particular month column w.r.t to date_report column
df['month'] = pd.to_datetime(df['date_report']).dt.month

# finding count of gender w.r.t to month column
infectors_month_count = df.groupby(['month'])['sex'].count()
print('total infected count per month: \n', infectors_month_count)

# finding male and female infectors count
female_infectors = df[df['sex'] == 'Female']
male_infectors = df[df['sex'] == 'Male']

# finding female and male infectors count w.r.t count
female_inf_count = female_infectors.groupby(['month'])['sex'].count()
male_inf_count = male_infectors.groupby(['month'])['sex'].count()
print('female_infected count per month:\n', female_inf_count)
print('male_infected count per month:\n', male_inf_count)



# 2bii)Sort the age groups with regards to the number of female infectors in descending order.
# sorting female infectors count
female_infectors_sort = female_infectors['age'].value_counts()
print(female_infectors_sort.sort_values(ascending=False))



# 2biii)For the person who does not has travel history, what are the top TWO (2) months with regards to the number of infectors older than 50?
#seperating columns who are having female travel history
no_travel_history = df[df['has_travel_history'] == 'f']
#considering columns only having >50 range
travel_hist_050 = no_travel_history[no_travel_history['age'].isin(['50-59', '60-69', '70-79', '80-89', '90-99'])]
#sorting age ranges w.r.t month
month_counts = travel_hist_050.groupby('month')['age'].count()
p = month_counts.sort_values(ascending=False)
#printing top 2 columns using head() statement
print(p.head(2))





# question3:
qdata = pd.read_csv('new_data.csv', index_col=0)
#creating a dataframe
df = pd.DataFrame(qdata)
# print(df)
df['month'] = pd.to_datetime(df['date_report']).dt.month

# 3a) Load the .csv file obtained in Q1(c) to Pandas Dataframe and derive the answer for the same 3 question in Q2(b).
# 3ai)What is the total number of male and female infectors for each month?

# finding count of gender w.r.t to month column
infectors_month_count = df.groupby(['month'])['sex'].count()
print('total infected count per month: \n', infectors_month_count)

# finding male and female infectors count
female_infectors = df[df['sex'] == 'Female']
male_infectors = df[df['sex'] == 'Male']

# finding female and male infectors count w.r.t count
female_inf_count = female_infectors.groupby(['month'])['sex'].count()
male_inf_count = male_infectors.groupby(['month'])['sex'].count()
print('female_infected count per month:\n', female_inf_count)
print('male_infected count per month:\n', male_inf_count)



# 3aii)Sort the age groups with regards to the number of female infectors in descending order.
# sorting female infectors count
female_infectors_sort = female_infectors['age'].value_counts()
print(female_infectors_sort.sort_values(ascending=False))



# 3aiii)For the person who does not has travel history, what are the top TWO (2) months with regards to the number of infectors older than 50?
#seperating columns who are having female travel history
no_travel_history = df[df['has_travel_history'] == 'f']
#considering columns only having >50 range
travel_hist_050 = no_travel_history[no_travel_history['age'].isin(['50-59', '60-69', '70-79', '80-89', '90-99'])]
#sorting age ranges w.r.t month
month_counts = travel_hist_050.groupby('month')['age'].count()
p = month_counts.sort_values(ascending=False)
#printing top 2 columns using head() statement
print(p.head(2))



# 3b1) Design a function to find out the top THREE (3) provinces of each month with regards to the total number of COVID-19 cases.
print('months with respective counts:\n', df['month'].value_counts())
#seperating age column w.r.t moht and province
province_with_high_cases = grouped = df.groupby(['month', 'province'])['age'].count()
print('provinces and months with their respective case count:\n',province_with_high_cases)
#listing top 3 provinces based on month criteria
top_provinces = province_with_high_cases.groupby('month', group_keys=False).nlargest(3)
print('top 3 provinces with cases on each month:\n', top_provinces)



# 3b2)Draw ONE (1) figure to show the total number of COVID-19 cases for each province.
df = pd.read_csv('new_data.csv', index_col=0)
# print(df['age'].value_counts())
a3 = df['province'].value_counts()
# print(a3)
#plotting a bar graph for each provinces bases on cases count
print('bar graph for total number of COVID-19 cases for each province')
print(a3.plot(kind='bar', xlabel='provinces', ylabel='cases count'))
plt.title('total number of COVID-19 cases for each province')
plt.show()



# 3b3)For the province with highest number of cases, draw ONE (1) figure to describe the age distributions of COVID-19 cases per gender.
df = pd.read_csv('new_data.csv', index_col=0)
#finding provice with highest no.of.cases
a4 = df['province'].value_counts().index[0]
#seperating data which are having the province equal to highest province
high_data = df[df['province'] == a4]
#seperating provinces count on the basis of age and gender
ages_in_province = high_data.groupby(['age', 'sex']).count()
#plotting a graph for provinces count calculated above
print(ages_in_province.plot(kind='bar', xlabel='age', ylabel='no.of cases'))
plt.title('age distributions of COVID-19 cases per gender for highest province')
plt.show()



# 3c1)Design a function to compute the days in a week (i.e. Mon, Tue or Wed) for any given date. Then update the column ‘report week’ in Dataframe by calling the function.
df['day'] = pd.to_datetime(df['report_week']).dt.day
# print(df.head())
# print(df['day'].value_counts())
#writing a function to define days w.r.t dates
def get_week(a1):
    d_name = {0: 'Mon', 1: 'Tue', 2: 'Wed',
              3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    return d_name[a1]

#converting the dates into dates number inorder to compute its corresponding day name
day = df['day']
for i in day:
    a = i % 7
    days = get_week(a)
    day.replace(to_replace=i, value=days, inplace=True)    
#reassigning the day column to report week column
df['report_week'] = df['day']
# print(day)
print(df['report_week'])



# 3c2) List out the top THREE (3) days that COVID-19 cases detected.
count = df['report_week'].value_counts()
#listing top 3 days using head() statement
print('top THREE (3) days that COVID-19 cases detected:\n', count.head(3))



# 3c3) Draw ONE (1) figure to describe the number of COVID-19 cases per gender for each month.
#creating month column w.r.t to date_report column
df['month'] = pd.to_datetime(df['date_report']).dt.month
print('dates to days:\n', df['month'])
#finding cases count w.r.t to month and gender by grouping them
cases_per_gender = df.groupby(['month', 'sex']).count()
print('cases_per_gender:\n', cases_per_gender)
print('number of COVID-19 cases per gender for each month')
#plotting graph for the cases count w.r.t to gender and month
cases_per_gender.plot(kind='bar', xlabel='Month', ylabel='no.of cases')
plt.title('COVID-19 Cases Per Gender for Each Month')
plt.legend(title='Gender')
plt.show()
