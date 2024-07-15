import streamlit as st

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

def main():
    # Dataframe selection
    st.markdown("<h1 align='center'> <b> Sistem Prediksi Limbah", unsafe_allow_html=True)
    new_line(1)
    st.markdown("Selamat datang! Sebuah aplikasi prediksi yang intuitif dan kuat yang dirancang untuk menyederhanakan proses membangun dan mengevaluasi model pembelajaran mesin. Sistem Prediksi ini menggunakan Algoritma ARIMA", unsafe_allow_html=True)
    
    st.divider()
    
    # Overview
    new_line()
    st.markdown("<h2 style='text-align: center;'>🗺️ Gambaran Umum</h2>", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""
    Ketika membangun model Prediksi, ada serangkaian langkah untuk menyiapkan data dan membangun model. Berikut ini adalah langkah-langkah utama dalam proses Machine Learning:
    
    - **📦 Pengumpulan Data**: proses pengumpulan data dari berbagai sumber seperti pustaka yfinance, file CSV, database, API, dll.<br> <br>
    - **🧹 Data Cleaning**: proses pembersihan data dengan menghapus duplikasi, menangani nilai yang hilang dll. Langkah ini sangat penting karena seringkali data tidak bersih dan mengandung banyak nilai yang hilang dan outlier. <br> <br>
    - **⚙️ Data Preprocessing**: proses mengubah data ke dalam format yang sesuai untuk analisis. Hal ini termasuk menangani fitur kategorikal, fitur numerik, penskalaan dan transformasi, dll. <br> <br>
    - **💡 Feature Engineering**: proses yang memanipulasi fitur itu sendiri. Terdiri dari beberapa langkah seperti ekstraksi fitur, transformasi fitur, dan pemilihan fitur. <br> <br>
    - **✂️ Splitting the Data**: proses membagi data menjadi set pelatihan, validasi, dan pengujian. Set pelatihan digunakan untuk melatih model, set validasi digunakan untuk menyetel hiperparameter, dan set pengujian digunakan untuk mengevaluasi model. <br> <br>
    - **🧠 Building Machine Learning Models**: Model yang digunakan pada aplikasi ini adalah ARIMA (AutoRegressive Integrated Moving Average). Model ARIMA sangat populer dalam analisis deret waktu untuk memprediksi data masa depan berdasarkan pola dari data historis. <br> <br>
    - **⚖️ Evaluating Machine Learning Models**: proses mengevaluasi model prediksi dengan menggunakan metrik seperti Mean Absolute Percentage Error (MAPE), Mean Squared Error (MSE), dan Root Mean Squared Error (RMSE). <br> <br>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Pada bagian membangun model, pengguna memasukkan nilai masing-masing hyperparameter. Hiperparameter adalah variabel yang secara signifikan mempengaruhi proses pelatihan model:
    
    - **⏱ P**: Adalah parameter berupa nilai integer yang menentukan jumlah lag pengamatan untuk model ARIMA. <br> <br>
    - **🧾 D**: Adalah jumlah diferensiasi yang diperlukan untuk membuat data menjadi stasioner. <br> <br>
    - **💣 Q**: Adalah jumlah lag dari komponen moving average. <br> <br>
    - **📚 Seasonal Order**: Parameter untuk menangani pola musiman dalam data. <br> <br>
    """, unsafe_allow_html=True)
    new_line()
    

    
    st.markdown("""Jika anda memiliki pertanyaan atau saran, jangan ragu untuk menghubungi Kami di sini untuk membantu!
    


Kami menantikan kabar dari Anda dan mendukung Anda dalam perjalanan pembelajaran mesin Anda!

    
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
