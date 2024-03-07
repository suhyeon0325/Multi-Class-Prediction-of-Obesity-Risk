import streamlit as st
import pandas as pd
from joblib import load
import os

# Assuming your model is saved in the 'model' directory with the name 'obesity_prediction_pipeline.joblib'
model_directory = 'model'
model_path = os.path.join(model_directory, 'NObeyesdad_prediction_pipeline.joblib')

def predict_NObeyesdad_level(model_path, Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS):
    
    # 모델 불러오기
    # RESTful API 방식으로 모델 호출
    pipeline = load(model_path)
    
    # 데이터프레임 생성
    df = pd.DataFrame([{
        'Gender': Gender, 'Age': Age, 'Height': Height, 'Weight': Weight,
        'family_history_with_overweight': family_history_with_overweight, 'FAVC': FAVC,
        'FCVC': FCVC, 'NCP': NCP, 'CAEC': CAEC, 'SMOKE': SMOKE, 'CH2O': CH2O,
        'SCC': SCC, 'FAF': FAF, 'TUE': TUE, 'CALC': CALC, 'MTRANS': MTRANS
    }])
    
    # 예측 값 생성
    prediction = pipeline.predict(df)
    return prediction[0]

def main():
    st.title('Obesity Level Prediction Model')
    st.write('Predict obesity levels based on personal and health-related attributes.')

    # Create input fields for each feature
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.number_input('Age', min_value=0.0, format='%f')
    Height = st.number_input('Height (in meters)', min_value=0.0, format='%f')
    Weight = st.number_input('Weight (in kg)', min_value=0.0, format='%f')
    family_history_with_overweight = st.selectbox('Family history with overweight', ['yes', 'no'])
    FAVC = st.selectbox('Frequent consumption of high caloric food', ['yes', 'no'])
    FCVC = st.number_input('Frequency of consumption of vegetables', min_value=0.0, max_value=3.0, step=0.1)
    NCP = st.number_input('Number of main meals', min_value=1.0, max_value=4.0, step=0.1)
    CAEC = st.selectbox('Consumption of food between meals', ['No', 'Sometimes', 'Frequently', 'Always'])
    SMOKE = st.selectbox('Do you smoke?', ['yes', 'no'])
    CH2O = st.number_input('Consumption of water daily (liters)', min_value=0.0, format='%f')
    SCC = st.selectbox('Calories consumption monitoring', ['yes', 'no'])
    FAF = st.number_input('Physical activity frequency (per week)', min_value=0.0, format='%f')
    TUE = st.number_input('Time using technology devices (hours)', min_value=0.0, format='%f')
    CALC = st.selectbox('Consumption of alcohol', ['Never', 'Sometimes', 'Frequently', 'Always'])
    MTRANS = st.selectbox('Mode of transportation', ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'])

    if st.button('Predict Obesity Level'):
        result = predict_NObeyesdad_level(model_path, Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS)
        st.success(f'Predicted Obesity Level: {result}')

if __name__ == "__main__":
    main()