import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

iris = pd.read_csv('iris.csv')
x = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

st.title('Iris Prediction Species')
st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', float(iris['sepal.length'].min()), float(iris['sepal.length'].max()), float(iris['sepal.length'].mean()))
    sepal_width = st.sidebar.slider('Sepal width', float(iris['sepal.width'].min()), float(iris['sepal.width'].max()), float(iris['sepal.width'].mean()))
    petal_length = st.sidebar.slider('Petal length', float(iris['petal.length'].min()), float(iris['petal.length'].max()), float(iris['petal.length'].mean()))
    petal_width = st.sidebar.slider('Petal width', float(iris['petal.width'].min()), float(iris['petal.width'].max()), float(iris['petal.width'].mean()))
    data = {'sepal.length': sepal_length,
            'sepal.width': sepal_width,
            'petal.length': petal_length,
            'petal.width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Parameters')
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('The predicted species is:', prediction)

st.subheader('Prediction Probability')
st.write('Probability of each species:')
st.write('Setosa:', prediction_proba[0][0])
st.write('Versicolor:', prediction_proba[0][1])
st.write('Virginica:', prediction_proba[0][2])

