import streamlit as st  
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load model
# model = pickle.load(open('diabetes_model.pkl', 'rb'))
model = pickle.load(open('Diabetes prediction/diabetes_model.pkl', 'rb'))

st.title('🩺 Diabetes Prediction App')

# Feature Importance section
st.subheader('📊 Feature Importance')
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
importances = model.feature_importances_
fig, ax = plt.subplots()
ax.barh(feature_names, importances)
st.pyplot(fig)

st.subheader('🔮 Make a Prediction')

# Input fields (all 8 features)
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
bp = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
skin = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input('Age', min_value=1, max_value=120, value=30)

# Predict
if st.button('Predict'):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error('⚠️ High risk of diabetes')
    else:
        st.success('✅ Low risk of diabetes')
