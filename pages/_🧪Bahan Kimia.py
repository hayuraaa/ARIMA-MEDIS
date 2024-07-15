import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import pmdarima as pm

# Title
st.title("Time Series Forecasting using ARIMA")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("DataFrame:", df)
    
    # Preprocess data
    df = df[['Produksi (Jumlah) Bahan Kimia']]
    
    # Plot the data
    st.subheader("Plot Data Original")
    st.line_chart(df)
    
    # Log transform the data
    df = np.log(df)
    st.subheader("Plot Transformasi Log Data")
    st.line_chart(df)
    
    # Train-test split
    msk = (df.index < len(df) / 3 * 2)
    df_train = df[msk].copy()
    df_test = df[~msk].copy()
    
    # Plot ACF and PACF for original data
    st.subheader("Plot ACF dan PACF untuk Data Original")
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df_train, ax=ax[0])
    ax[0].set_title("Autocorrelation Function (ACF)")
    plot_pacf(df_train, lags=11, ax=ax[1])
    ax[1].set_title("Partial Autocorrelation Function (PACF)")
    st.pyplot(fig)
    
    # ADF test for stationarity
    adf_test = adfuller(df_train)
    st.subheader("ADF Test untuk Keberhentian Data Original")
    st.write(f'p-value: {adf_test[1]}')
    
    # Differencing the data
    df_train_diff = df_train.diff().dropna()
    st.subheader("Plot Data yang Sudah Di-Differensiasi")
    st.line_chart(df_train_diff)
    
    # Plot ACF and PACF for differenced data
    st.subheader("Plot ACF dan PACF untuk Data yang Sudah Di-Differensiasi")
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df_train_diff, ax=ax[0])
    ax[0].set_title("Autocorrelation Function (ACF) untuk Data Differensiasi")
    plot_pacf(df_train_diff, lags=10, ax=ax[1])
    ax[1].set_title("Partial Autocorrelation Function (PACF) untuk Data Differensiasi")
    st.pyplot(fig)
    
    # ADF test for differenced data
    adf_test = adfuller(df_train_diff)
    st.subheader("ADF Test untuk Data yang Sudah Di-Differensiasi")
    st.write(f'p-value: {adf_test[1]}')
    
    # Manual ARIMA model fitting
    st.sidebar.subheader("Manual ARIMA Parameters")
    p_manual = st.sidebar.number_input("AR order (p)", min_value=0, step=1, value=1, key="p_manual")
    d_manual = st.sidebar.number_input("Differencing order (d)", min_value=0, step=1, value=1, key="d_manual")
    q_manual = st.sidebar.number_input("MA order (q)", min_value=0, step=1, value=1, key="q_manual")
    
    model_manual = ARIMA(df_train, order=(p_manual, d_manual, q_manual))
    model_manual_fit = model_manual.fit()
    st.subheader("Ringkasan Model ARIMA Manual")
    st.text(model_manual_fit.summary())
    
    # Residual plots
    residuals = model_manual_fit.resid[1:]
    st.subheader("Plot Residual dari Model ARIMA Manual")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    residuals.plot(title='Residuals', ax=ax[0])
    residuals.plot(title='Density', kind='kde', ax=ax[1])
    st.pyplot(fig)
    
    # Plot ACF and PACF for residuals
    st.subheader("Plot ACF dan PACF untuk Residual")
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(residuals, ax=ax[0])
    ax[0].set_title("Autocorrelation Function (ACF) untuk Residual")
    plot_pacf(residuals, lags=10, ax=ax[1])
    ax[1].set_title("Partial Autocorrelation Function (PACF) untuk Residual")
    st.pyplot(fig)
    
    # Forecast with manual ARIMA
    forecast_test_manual = model_manual_fit.forecast(len(df_test))
    df['forecast_manual'] = [None] * len(df_train) + list(forecast_test_manual)
    st.subheader("Plot Forecast dengan ARIMA Manual")
    st.line_chart(df)
    
    # Auto ARIMA model fitting
    st.sidebar.subheader("Auto ARIMA Parameters")
    stepwise = st.sidebar.checkbox("Stepwise", value=False, key="stepwise")
    seasonal = st.sidebar.checkbox("Seasonal", value=False, key="seasonal")
    
    auto_arima = pm.auto_arima(df_train, stepwise=stepwise, seasonal=seasonal)
    st.subheader("Ringkasan Model Auto ARIMA")
    st.text(auto_arima.summary())
    
    # Forecast with auto ARIMA
    forecast_test_auto = auto_arima.predict(n_periods=len(df_test))
    df['forecast_auto'] = [None] * len(df_train) + list(forecast_test_auto)
    st.subheader("Plot Forecast dengan Auto ARIMA")
    st.line_chart(df)
    
    st.write("Forecasts:", df)
    
    # Future forecasting
    forecast_periods = st.sidebar.number_input("Masukkan Jumlah Periode untuk Prediksi Masa Depan", min_value=1, step=1, value=12, key="forecast_periods")
    
    future_forecast_manual = model_manual_fit.forecast(len(df_test) + forecast_periods)[-forecast_periods:]
    future_forecast_auto = auto_arima.predict(n_periods=len(df_test) + forecast_periods)[-forecast_periods:]
    
    df_future = pd.DataFrame({
        'Produksi (Jumlah) Bahan Kimia': [None] * len(df) + [None] * forecast_periods,
        'forecast_manual': list(df['forecast_manual']) + list(future_forecast_manual),
        'forecast_auto': list(df['forecast_auto']) + list(future_forecast_auto)
    })
    
    st.subheader("Plot Forecast Masa Depan dengan Manual ARIMA dan Auto ARIMA")
    plt.figure(figsize=(10, 6))
    plt.plot(df['Produksi (Jumlah) Bahan Kimia'], label='Actual')
    plt.plot(df['forecast_manual'], label='Manual Forecast')
    plt.plot(df['forecast_auto'], label='Auto Forecast')
    plt.plot(df_future['forecast_manual'], label='Future Manual Forecast', linestyle='--')
    plt.plot(df_future['forecast_auto'], label='Future Auto Forecast', linestyle='--')
    plt.legend()
    st.pyplot(plt)
    
    # Error Metrics
    mae_manual = mean_absolute_error(df_test, forecast_test_manual)
    mape_manual = mean_absolute_percentage_error(df_test, forecast_test_manual)
    rmse_manual = np.sqrt(mean_squared_error(df_test, forecast_test_manual))
    
    mae_auto = mean_absolute_error(df_test, forecast_test_auto)
    mape_auto = mean_absolute_percentage_error(df_test, forecast_test_auto)
    rmse_auto = np.sqrt(mean_squared_error(df_test, forecast_test_auto))
    
    st.subheader("Error Metrics untuk Model ARIMA Manual dan Auto ARIMA")
    st.write("Manual ARIMA - MAE:", mae_manual, "MAPE:", mape_manual, "RMSE:", rmse_manual)
    st.write("Auto ARIMA - MAE:", mae_auto, "MAPE:", mape_auto, "RMSE:", rmse_auto)
