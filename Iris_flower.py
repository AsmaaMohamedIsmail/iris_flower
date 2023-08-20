import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
iris=datasets.load_iris()
x=iris.data
y=iris.target

rfc=RandomForestClassifier()
rfc.fit(x,y)

# streamlit structure:
st.write("Simple Prediction **Iris Flower** App!")

st.sidebar.header("Input Parameters")
def User_Inputs():
       first=st.sidebar.slider("Sepal_Lenght",4.30,7.90,5.40)
       second=st.sidebar.slider("Sepal_Width",2.5,4.4,3.4)
       third=st.sidebar.slider("Petal_Lenght",1.0,6.9,1.3)
       forth=st.sidebar.slider("Petal_Width",0.1,2.5,0.2)
       data={"Sepal_Lenght":first,
            "Sepal_Width":second,
            "Petal_Lenght":third,
            "Petal_Width":forth}
       features=pd.DataFrame(data,index=[0])

       return features
df=User_Inputs()

st.subheader("User Inputs Parameters")
st.write(df)

st.subheader("Target Names")
st.write(iris.target_names)

pred=rfc.predict(df)
pro=rfc.predict_proba(df)

st.subheader("Prediction")
st.write(iris.target_names[pred])

st.subheader("Prediction_probabilities")
st.write(pro)
