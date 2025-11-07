import os
import streamlit as st
import pandas as pd
import pickle

# ------------------------------------------------------------------
# 🔒 Safe path handling (important for Streamlit Cloud)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    """Safely load a pickle file from the app directory."""
    path = os.path.join(BASE_DIR, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)

# ------------------------------------------------------------------
# 📦 Load model, scaler, and metadata
# ------------------------------------------------------------------
model = load_pickle('attrition_model.pkl')
scaler = load_pickle('scaler.pkl')
feature_names = load_pickle('feature_names.pkl')
defaults = load_pickle('defaults.pkl')

# ------------------------------------------------------------------
# 🏷️ Streamlit App Interface
# ------------------------------------------------------------------
st.title('💼 HR Attrition Predictor')
st.write('Predict if an employee is likely to leave the company.')

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
    job_role = st.selectbox(
        'Job Role',
        [
            'Sales Executive', 'Research Scientist', 'Laboratory Technician',
            'Manufacturing Director', 'Healthcare Representative', 'Manager',
            'Sales Representative', 'Research Director', 'Human Resources'
        ]
    )

# ------------------------------------------------------------------
# 🔮 Prediction logic
# ------------------------------------------------------------------
if st.button('Predict Attrition'):
    # Create input dictionary from defaults
    input_dict = defaults.copy()

    # Update with user inputs
    input_dict.update({
        'Age': age,
        'MonthlyIncome': monthly_income,
        'YearsAtCompany': years_at_company,
        'TotalWorkingYears': total_working_years,
        'JobSatisfaction': job_satisfaction,
        'YearsSinceLastPromotion': years_since_promotion,
        'EnvironmentSatisfaction': environment_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'OverTime': 1 if overtime == 'Yes' else 0,
        'Gender': 1 if gender == 'Male' else 0
    })

    # Encode Department (one-hot)
    input_dict[f"Department_{department}"] = 1

    # Encode Job Role (one-hot)
    input_dict[f"JobRole_{job_role}"] = 1

    # Convert to DataFrame and align with training features
    input_df = pd.DataFrame([input_dict])[feature_names]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # ------------------------------------------------------------------
    # 🧾 Display result
    # ------------------------------------------------------------------
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

