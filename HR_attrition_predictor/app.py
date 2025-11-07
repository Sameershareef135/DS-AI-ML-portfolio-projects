import os
import streamlit as st
import pandas as pd
import pickle

# Get the directory where the app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, scaler, and feature names safely using absolute paths
model = pickle.load(open(os.path.join(BASE_DIR, 'attrition_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))
feature_names = pickle.load(open(os.path.join(BASE_DIR, 'feature_names.pkl'), 'rb'))
defaults = pickle.load(open(os.path.join(BASE_DIR, 'defaults.pkl'), 'rb'))


st.title('HR Attrition Predictor')
st.write('Predict if an employee will leave')

st.subheader('Enter Employee Details')

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', 18, 60, 30)
    monthly_income = st.number_input('Monthly Income', 1000, 20000, 5000)
    years_at_company = st.number_input('Years at Company', 0, 40, 5)
    total_working_years = st.number_input('Total Working Years', 0, 40, 10)
    overtime = st.selectbox('OverTime', ['No', 'Yes'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    
with col2:
    job_satisfaction = st.slider('Job Satisfaction', 1, 4, 3)
    years_since_promotion = st.number_input('Years Since Last Promotion', 0, 15, 1)
    environment_satisfaction = st.slider('Environment Satisfaction', 1, 4, 3)
    work_life_balance = st.slider('Work Life Balance', 1, 4, 3)
    department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
    job_role = st.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                                         'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                         'Sales Representative', 'Research Director', 'Human Resources'])

if st.button('Predict Attrition'):
    # Create dict with defaults (0 for all features)
    defaults = pickle.load(open('defaults.pkl', 'rb'))
    input_dict = defaults.copy()
    
    # Update with user inputs - numerical features
    input_dict['Age'] = age
    input_dict['MonthlyIncome'] = monthly_income
    input_dict['YearsAtCompany'] = years_at_company
    input_dict['TotalWorkingYears'] = total_working_years
    input_dict['JobSatisfaction'] = job_satisfaction
    input_dict['YearsSinceLastPromotion'] = years_since_promotion
    input_dict['EnvironmentSatisfaction'] = environment_satisfaction
    input_dict['WorkLifeBalance'] = work_life_balance
    
    # Encode OverTime
    input_dict['OverTime'] = 1 if overtime == 'Yes' else 0
    
    # Encode Gender
    input_dict['Gender'] = 1 if gender == 'Male' else 0
    
    # Encode Department (one-hot)
    if department == 'Sales':
        input_dict['Department_Sales'] = 1
    elif department == 'Research & Development':
        input_dict['Department_Research & Development'] = 1
    elif department == 'Human Resources':
        input_dict['Department_Human Resources'] = 1
    
    # Encode JobRole (one-hot)
    if job_role == 'Sales Executive':
        input_dict['JobRole_Sales Executive'] = 1
    elif job_role == 'Research Scientist':
        input_dict['JobRole_Research Scientist'] = 1
    elif job_role == 'Laboratory Technician':
        input_dict['JobRole_Laboratory Technician'] = 1
    elif job_role == 'Manufacturing Director':
        input_dict['JobRole_Manufacturing Director'] = 1
    elif job_role == 'Healthcare Representative':
        input_dict['JobRole_Healthcare Representative'] = 1
    elif job_role == 'Manager':
        input_dict['JobRole_Manager'] = 1
    elif job_role == 'Sales Representative':
        input_dict['JobRole_Sales Representative'] = 1
    elif job_role == 'Research Director':
        input_dict['JobRole_Research Director'] = 1
    elif job_role == 'Human Resources':
        input_dict['JobRole_Human Resources'] = 1
    
    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])
    
    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display result
    st.subheader('Prediction Result')
    if prediction == 1:
        st.error(f'⚠️ **High Risk**: Employee likely to leave ({probability[1]*100:.1f}% probability)')
        st.write('**Recommended Actions:**')
        st.write('- Schedule retention conversation')
        st.write('- Review compensation and benefits')
        st.write('- Discuss career growth opportunities')
    else:
        st.success(f'✅ **Low Risk**: Employee likely to stay ({probability[0]*100:.1f}% probability)')
        st.write('Continue regular engagement and development activities.')
