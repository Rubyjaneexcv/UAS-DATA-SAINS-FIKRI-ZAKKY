import streamlit as st
import pandas as pd
import pickle

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Attrition Karyawan",
    page_icon="👨‍💼",
    layout="centered"
)

# --- Judul dan Deskripsi ---
st.title('👨‍💼 Aplikasi Prediksi Attrition Karyawan')
st.write("""
Aplikasi ini memprediksi kemungkinan seorang karyawan akan keluar dari perusahaan (attrition)
berdasarkan data yang Anda masukkan. Harap isi semua field di sidebar.
""")

# --- Muat Model dan Kolom ---
try:
    with open('model_rf.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model_columns.pkl', 'rb') as file:
        model_columns = pickle.load(file)
except FileNotFoundError:
    st.error("File model atau kolom tidak ditemukan. Pastikan 'model_rf.pkl' dan 'model_columns.pkl' berada di direktori yang sama.")
    st.stop()

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header('Masukkan Data Karyawan')

def user_input_features():
    Age = st.sidebar.slider('Usia', 18, 60, 35)
    DailyRate = st.sidebar.slider('Gaji Harian ($)', 100, 1500, 750)
    DistanceFromHome = st.sidebar.slider('Jarak dari Rumah (km)', 1, 30, 10)
    EnvironmentSatisfaction = st.sidebar.selectbox('Kepuasan Lingkungan', (1, 2, 3, 4))
    JobInvolvement = st.sidebar.selectbox('Keterlibatan Pekerjaan', (1, 2, 3, 4))
    JobLevel = st.sidebar.selectbox('Level Pekerjaan', (1, 2, 3, 4, 5))
    JobSatisfaction = st.sidebar.selectbox('Kepuasan Pekerjaan', (1, 2, 3, 4))
    MonthlyIncome = st.sidebar.slider('Pendapatan Bulanan ($)', 1000, 20000, 6500)
    NumCompaniesWorked = st.sidebar.slider('Jumlah Perusahaan Sebelumnya', 0, 9, 2)
    OverTime = st.sidebar.selectbox('Lembur (Overtime)', ('Yes', 'No'))
    PercentSalaryHike = st.sidebar.slider('Kenaikan Gaji (%)', 11, 25, 15)
    TotalWorkingYears = st.sidebar.slider('Total Tahun Bekerja', 0, 40, 10)
    YearsAtCompany = st.sidebar.slider('Tahun di Perusahaan Ini', 0, 40, 5)
    YearsInCurrentRole = st.sidebar.slider('Tahun di Posisi Saat Ini', 0, 18, 4)
    YearsSinceLastPromotion = st.sidebar.slider('Tahun Sejak Promosi Terakhir', 0, 15, 2)
    YearsWithCurrManager = st.sidebar.slider('Tahun dengan Manajer Saat Ini', 0, 17, 4)

    data = {
        'Age': Age, 'DailyRate': DailyRate, 'DistanceFromHome': DistanceFromHome,
        'EnvironmentSatisfaction': EnvironmentSatisfaction, 'JobInvolvement': JobInvolvement,
        'JobLevel': JobLevel, 'JobSatisfaction': JobSatisfaction, 'MonthlyIncome': MonthlyIncome,
        'NumCompaniesWorked': NumCompaniesWorked, 'PercentSalaryHike': PercentSalaryHike,
        'TotalWorkingYears': TotalWorkingYears, 'YearsAtCompany': YearsAtCompany,
        'YearsInCurrentRole': YearsInCurrentRole, 'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager, 'OverTime_Yes': 1 if OverTime == 'Yes' else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Tombol Prediksi ---
if st.button('🔮 Prediksi Sekarang'):
    # Gabungkan input dengan DataFrame kosong yang memiliki semua kolom model
    final_df = pd.DataFrame(columns=model_columns)
    # Gunakan pd.concat bukan append
    final_df = pd.concat([final_df, input_df], ignore_index=True).fillna(0)

    # Pastikan urutan kolom input sama persis dengan saat training
    final_df = final_df[model_columns]

    # Lakukan prediksi
    prediction = model.predict(final_df)
    prediction_proba = model.predict_proba(final_df)

    st.subheader('Hasil Prediksi:')
    if prediction[0] == 1:
        st.error(f'**Karyawan Berisiko Tinggi untuk Attrition (Keluar)**')
        st.write(f"Probabilitas Attrition: **{prediction_proba[0][1]*100:.2f}%**")
    else:
        st.success(f'**Karyawan Cenderung Bertahan di Perusahaan**')
        st.write(f"Probabilitas Bertahan: **{prediction_proba[0][0]*100:.2f}%**")