import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def generate_wind_turbine_data():
    # Generate sample data for wind turbine components
    np.random.seed(0)
    timestamps = pd.date_range(start='2022-01-01', periods=100, freq='H')
    gearbox_temperature = np.random.normal(50, 5, 100)
    blade_vibration = np.random.normal(0.5, 0.1, 100)
    generator_voltage = np.random.normal(400, 20, 100)
    yaw_position = np.random.normal(180, 10, 100)
    main_shaft_speed = np.random.normal(10, 1, 100)
    
    df = pd.DataFrame({'Timestamp': timestamps,
                       'Gearbox Temperature': gearbox_temperature,
                       'Blade Vibration': blade_vibration,
                       'Generator Voltage': generator_voltage,
                       'Yaw Position': yaw_position,
                       'Main Shaft Speed': main_shaft_speed})
    return df

def anomaly_detection(df, contamination=0.1):
    # Anomaly detection for each component
    anomaly_results = {}
    components = df.columns[1:]  # Exclude Timestamp
    for component in components:
        model = IsolationForest(contamination=contamination)
        df['anomaly'] = model.fit_predict(df[[component]])
        anomaly_results[component] = df[df['anomaly'] == -1]

    return anomaly_results

def plot_anomaly_graphs(anomaly_results):
    # Plot anomaly graphs for each component
    for component, anomaly_df in anomaly_results.items():
        fig, ax = plt.subplots()
        ax.plot(anomaly_df['Timestamp'], anomaly_df[component], 'ro', label='Anomaly')
        ax.plot(anomaly_df['Timestamp'], anomaly_df[component], 'k--', alpha=0.5)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel(component)
        ax.set_title(f'Anomalies in {component}')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

def main():
    st.title('Wind Turbine Anomaly Detection')

    # File upload
    st.sidebar.subheader('Upload Data')
    uploaded_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'])

    if uploaded_file is not None:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)
    else:
        # Generate wind turbine data
        df = generate_wind_turbine_data()

    # Display data table
    st.subheader('Wind Turbine Data')
    st.write(df)

    # Slider for contamination parameter
    contamination = st.sidebar.slider('RPM', min_value=0.0, max_value=0.5, step=0.01, value=0.1)

    # Perform anomaly detection
    anomaly_results = anomaly_detection(df, contamination=contamination)

    # Plot anomaly graphs for each component
    plot_anomaly_graphs(anomaly_results)

if __name__ == '__main__':
    main()
