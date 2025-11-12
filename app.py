
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Streamlit app title
st.title('Prediksi Tagihan Listrik Jakarta')
st.write('Aplikasi untuk memprediksi jumlah tagihan listrik berdasarkan parameter yang diberikan.')

# Sidebar for user inputs
st.sidebar.header('Input Parameter')

def user_input_features():
    kwh = st.sidebar.slider('Konsumsi KWH (kWh)', 150.0, 600.0, 350.0)
    ac_units = st.sidebar.slider('Jumlah AC', 0, 3, 1)
    ac_hours_per_day = st.sidebar.slider('Jam AC per Hari', 0.0, 10.0, 5.0)
    family_size = st.sidebar.slider('Jumlah Anggota Keluarga', 2, 6, 4)
    
    month_name = st.sidebar.selectbox('Bulan', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    tariff_class = st.sidebar.selectbox('Kelas Tarif', ['R1', 'R2', 'R3'])
    
    data = {
        'kwh': kwh,
        'ac_units': ac_units,
        'ac_hours_per_day': ac_hours_per_day,
        'family_size': family_size,
        'month_name': month_name,
        'tariff_class': tariff_class
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Parameter Input Pengguna:')
st.write(df_input)

# One-hot encode categorical features, matching the training data columns
# List of all columns used during training (extracted from X_train.columns)
# The order of these columns is crucial for the model prediction
training_columns = [
    'kwh', 'ac_units', 'ac_hours_per_day', 'family_size', 
    'month_name_Aug', 'month_name_Dec', 'month_name_Feb', 'month_name_Jan', 
    'month_name_Jul', 'month_name_Jun', 'month_name_Mar', 'month_name_May', 
    'month_name_Nov', 'month_name_Oct', 'month_name_Sep', 
    'tariff_class_R2', 'tariff_class_R3'
]

# Create a DataFrame with all training columns, initialized to 0
final_input = pd.DataFrame(0, index=[0], columns=training_columns)

# Populate numerical features
final_input['kwh'] = df_input['kwh'][0]
final_input['ac_units'] = df_input['ac_units'][0]
final_input['ac_hours_per_day'] = df_input['ac_hours_per_day'][0]
final_input['family_size'] = df_input['family_size'][0]

# Populate one-hot encoded categorical features
if f"month_name_{df_input['month_name'][0]}" in final_input.columns:
    final_input[f"month_name_{df_input['month_name'][0]}"] = 1

if f"tariff_class_{df_input['tariff_class'][0]}" in final_input.columns:
    final_input[f"tariff_class_{df_input['tariff_class'][0]}"] = 1


# Ensure the order of columns in final_input matches the model's expected input
final_input = final_input[training_columns]

# Make prediction
if st.sidebar.button('Prediksi Tagihan'):
    prediction = model.predict(final_input)
    st.subheader('Hasil Prediksi Tagihan Listrik:')
    st.write(f"Tagihan Diprediksi: Rp {prediction[0]:,.2f}")

