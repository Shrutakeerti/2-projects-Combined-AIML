import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open(r'D:\Jal Shakti\well_extraction_model.pkl', 'rb') as f:
    model = pickle.load(f)


st.title("Water Well Suitability Prediction")
st.write("This app predicts if a location is suitable for high groundwater extraction based on environmental features.")


Recharge_from_rainfall_during_Monsoon_Season = st.number_input("Recharge_from_rainfall_during_Monsoon_Season (mm):", min_value=0.0, max_value=200000.0)
Recharge_from_Other_Resources = st.number_input("Recharge_from_Other_Resources(in MCM):", min_value=0.0, max_value=10000.0)
Recharge_from_Rainfall_NonMonsoon = st.number_input("Recharge_from_Rainfall_NonMonsoon (in MCM):", min_value=0.0, max_value=100000.0)
Recharge_sources_NonMonsoon = st.number_input("Recharge_sources_NonMonsoon(in MCM):", min_value=0.0, max_value=1000000.0)
groundwater_avail = st.number_input("groundwater_avail(in MCM):", min_value=0.0, max_value=1000000.0)


Total_Natural_Discharges = st.number_input("Total_Natural_Discharges:", min_value=0.0, max_value=1000000.0)
Annual_Extractable_Ground_Water = st.number_input("Annual_Extractable_Ground_Water:", min_value=0.0, max_value=1000000.0)
Current_Annual_Ground_Water_for_Irrigation = st.number_input("Current_Annual_Ground_Water_for_Irrigation:", min_value=0.0, max_value=1000000.0)
Total_Currrent_Annual_Ground_Water = st.number_input("Total_Currrent_Annual_Ground_Water:", min_value=0.0, max_value=1000000.0)
Annual_GW_Allocation_for_Domestic_Use = st.number_input("Annual_GW_Allocation_for_Domestic_Use:", min_value=0.0, max_value=1000000.0)
Net_Ground_Water_future_use = st.number_input("Net_Ground_Water_future_use:", min_value=0.0, max_value=1000000.0)
Stage_ground_water_extraction = st.number_input("Stage_ground_water_extraction:", min_value=0.0, max_value=1000000.0)


input_data = np.array([[Recharge_from_rainfall_during_Monsoon_Season, Recharge_from_Other_Resources, Recharge_from_Rainfall_NonMonsoon, Recharge_sources_NonMonsoon, groundwater_avail,
                        Total_Natural_Discharges, Annual_Extractable_Ground_Water, Current_Annual_Ground_Water_for_Irrigation, Total_Currrent_Annual_Ground_Water, Annual_GW_Allocation_for_Domestic_Use, Net_Ground_Water_future_use, Stage_ground_water_extraction]])


if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "High Extraction Suitability" if prediction[0] == 1 else "Low Extraction Suitability"
    st.write(f"Prediction: {result}")
