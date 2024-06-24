import streamlit as st
import pandas as pd
import pickle
import os

# Function to load the best model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'best_heart_disease_model.pkl')
    
    model = pickle.load(open(model_path, 'rb'))
    
    return model

# Function to preprocess input data
def preprocess_input(input_data):
    # Implement your preprocessing logic here
    # Make sure it matches the preprocessing done during model training
    
    # Example: Transform categorical variables using the same logic used in Q3.2
    # Note: Ensure categorical encoding matches the encoding used in Q3.2
    # You may need to include encoding for categorical variables like 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
    input_data['sex'] = input_data['sex'].apply(lambda x: 1 if x == 'Male' else 0)
    input_data['cp'] = input_data['cp'].apply(lambda x: 3 if x == 'Typical Angina' else (2 if x == 'Atypical Angina' else (1 if x == 'Non-anginal Pain' else 0)))
    input_data['fbs'] = input_data['fbs'].apply(lambda x: 1 if x == 'True' else 0)
    input_data['restecg'] = input_data['restecg'].apply(lambda x: 0 if x == 'Normal' else (1 if x == 'ST-T wave abnormality' else 2))
    input_data['exang'] = input_data['exang'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['slope'] = input_data['slope'].apply(lambda x: 0 if x == 'Upsloping' else (1 if x == 'Flat' else 2))
    input_data['thal'] = input_data['thal'].apply(lambda x: 2 if x == 'Reversable Defect' else (1 if x == 'Fixed Defect' else 0))
    
    return input_data

def main():
    st.title("Heart Disease Prediction")
    st.sidebar.title("User Input")
    
    # Load model
    model = load_model()
    
    # Define user input form
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", ["0", "1", "2", "3", "4"])
    thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])
    
    # Prepare input data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # Preprocess input data
    input_data = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction
    st.write("## Prediction")
    if prediction[0] == 1:
        st.write("The patient is likely to have heart disease.")
    else:
        st.write("The patient is unlikely to have heart disease.")

if __name__ == '__main__':
    main()
