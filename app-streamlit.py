import streamlit as st
import joblib
import pandas as pd

# Load the model and pipeline
model = joblib.load('model.pkl')

# Set the title of the app
st.title("Titanic Survival Prediction")

# Add a description
st.write("""
This app predicts whether a passenger would have survived the Titanic disaster based on their features.
""")

# Input fields for user data
st.sidebar.header("User Input Features")

def user_input_features():
    pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    age = st.sidebar.slider("Age", 0.0, 100.0, 30.0)
    sibsp = st.sidebar.slider("SibSp (Number of Siblings/Spouses Aboard)", 0, 8, 0)
    parch = st.sidebar.slider("Parch (Number of Parents/Children Aboard)", 0, 6, 0)
    fare = st.sidebar.slider("Fare", 0.0, 600.0, 32.0)
    embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

    # Create a dictionary from the input
    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }

    # Convert the dictionary into a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display the user input
st.subheader("User Input Features")
st.write(input_df)

# Make a prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display the prediction
st.subheader("Prediction")
survival_status = "Survived" if prediction[0] == 1 else "Not Survived"
st.write(f"The passenger would have **{survival_status}**.")

# Display prediction probabilities
st.subheader("Prediction Probability")
st.write(f"Probability of Survival: {prediction_proba[0][1]:.2f}")
st.write(f"Probability of Not Surviving: {prediction_proba[0][0]:.2f}")