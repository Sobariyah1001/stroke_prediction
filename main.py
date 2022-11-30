import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree



st.write("""
#Aplikasi cek stroke
Ini adalah aplikasi untuk mengecek kesehatan anda mengenai stroke

""")

df = pd.read_csv("https://raw.githubusercontent.com/Sobariyah1001/dataset/main/healthcare-dataset-stroke-data.csv")

desc = st.button("Dataset")
proc = st.button("Processing")
model = st.button("Model")
cek = st.button("Cek")

#READ DATA
df = pd.read_csv("https://raw.githubusercontent.com/Sobariyah1001/dataset/main/healthcare-dataset-stroke-data.csv")

#EKSPLORE DATA
df[["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]].agg(['min','max'])
df.stroke.value_counts()

#POCESSING DATA
df = df.drop(columns="id")
X = df.drop(columns="stroke")
y = df.stroke
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
le.inverse_transform(y)
labels = pd.get_dummies(df.stroke).columns.values.tolist()
labels

#NORMALISASI
dataubah=df.drop(columns=['gender','ever_married','work_type','Residence_type','smoking_status'])

data_gen=df[['gender']]
gen = pd.get_dummies(data_gen)

data_married=df[['ever_married']]
married = pd.get_dummies(data_married)

data_work=df[['work_type']]
work = pd.get_dummies(data_work)

data_residence=df[['Residence_type']]
residence = pd.get_dummies(data_residence)

data_smoke=df[['smoking_status']]
smoke = pd.get_dummies(data_smoke)

data_bmi = df[['bmi']]
bmi = pd.get_dummies(data_bmi)

dataOlah = pd.concat([gen,married,work,residence,smoke,bmi], axis=1)
dataHasil = pd.concat([df,dataOlah], axis = 1)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X.shape, y.shape
le.inverse_transform(y)
labels = pd.get_dummies(dataHasil.stroke).columns.values.tolist()
labels

#NORMALISASI MinMax Scaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X.shape, y.shape

#SPLIT DATA
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#MODEL
#knn
metode1 = KNeighborsClassifier(n_neighbors=3)
metode1.fit(X_train, y_train)
print(metode1.score(X_train, y_train))
print(metode1.score(X_test, y_test))
y_pred = metode1.predict(scaler.transform(array([[50.0,0,1,105.92,0,0,1,0,1,0,1,1,1,1,1,1,1,0,0,0,0]])))
le.inverse_transform(y_pred)[0]

#GAUSSIAN NAIVE BAYES
metode2 = GaussianNB()
metode2.fit(X_train, y_train)
print(metode2.score(X_train, y_train))
print(metode2.score(X_test, y_test))
y_pred = metode2.predict(array([[50.0,0,1,105.92,0,0,1,0,1,0,1,1,1,1,1,1,1,0,0,0,0]]))
le.inverse_transform(y_pred)[0]

#DECISION TREE
metode3 = tree.DecisionTreeClassifier(criterion="gini")
metode3.fit(X_train, y_train)
print(metode3.score(X_train, y_train))
print(metode3.score(X_test, y_test))
y_pred = metode3.predict(array([[50.0,0,1,105.92,0,0,1,0,1,0,1,1,1,1,1,1,1,0,0,0,0]]))
le.inverse_transform(y_pred)[0]

if desc:
    ## Exploration Data
    st.write("Data yang digunakan adalah stroke dataset yang diambil dari : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset")
    st.write("Jumlah dataset adalah 5110 dengan")
    st.write("Data Training = 80%")
    
    X_train
    y_train

    st.write("Data Testing = 20%")
    X_test
    y_test

    st.write("Berikut adalah data yang digunakan untuk memprediksi apakah seseorang terserang stroke atau tidak dengan beberapa fitur")
    df
if proc:
    st.success(f"Hello")