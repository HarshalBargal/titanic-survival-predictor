import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load your trained model
model = pickle.load(open('titanic_model.pkl', 'rb'))

# Load dataset for plots
df = pd.read_csv('train.csv')

# Fill missing values for plots
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['HasCabin'] = df['Cabin'].notnull().astype(int)

# App title
st.title("🚢 Titanic Survival Predictor + Visualizations")

# User Inputs
st.header("Passenger Details")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 25)
HasCabin = st.selectbox("Has Cabin?", ["No", "Yes"])
FamilySize = st.slider("Family Size", 1, 10, 1)
Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

Sex = 0 if Sex == "male" else 1
HasCabin = 0 if HasCabin == "No" else 1
Embarked_C = 1 if Embarked == "C" else 0
Embarked_Q = 1 if Embarked == "Q" else 0

# Prediction
if st.button("Predict Survival"):
    result = model.predict([[Pclass, Sex, Age, HasCabin, FamilySize, Embarked_C, Embarked_Q]])
    st.write("✅ Survived" if result[0] == 1 else "❌ Did Not Survive")

# Visualizations
st.header("Visualizations")

if st.checkbox("Show Survival Countplot"):
    fig1 = plt.figure()
    sns.countplot(x='Survived', data=df)
    plt.title('Survival Count')
    st.pyplot(fig1)

if st.checkbox("Show Survival by Pclass"):
    fig2 = plt.figure()
    sns.countplot(x='Pclass', hue='Survived', data=df)
    plt.title('Survival by Passenger Class')
    st.pyplot(fig2)

if st.checkbox("Show Survival Pie Chart"):
    survived = df['Survived'].value_counts()
    fig3 = plt.figure()
    plt.pie(survived, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=90)
    plt.title('Survival Pie Chart')
    st.pyplot(fig3)
