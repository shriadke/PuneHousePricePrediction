import streamlit as st
from housePricePrediction.pipeline.prediction import PredictionPipeline
from housePricePrediction.utils.data_utils import get_area_idx_dict

import pandas as pd
import string

st.title('Welcome to Price Prediction App')
st.title('🔎 _Enter_ Property Details!')
st.markdown("""Enter as much accurate data as possible, data validation is not performed for this app!""")

columns=["Location","Type","Area_sqft", "Developer", "Name", "Area_township", "hasClubHouse_yes", "hasEduFacility_yes", "hasHospital_yes", "hasMall_yes", "hasParkOrJogTrack_yes", "hasPool_yes", "hasGym_yes"]
options = ["No", "Yes"]
sub_area_list =[string.capwords(ar) for ar in get_area_idx_dict().keys()]
sub_area_list.append("Other")


with st.form("search_form"):
    col1, col2 = st.columns(2)
    with col1:
        loc = st.text_input("Enter Location", key="loc", value="Pune, MH, India")
        # sub_Area = st.text_input("Enter Sub-Area*:", key="sub_area", placeholder = "Sub Area")
        prop_type = st.text_input("Enter Property Type", key="prop_type",placeholder="1 BHK, 2 BHK, etc.")
        comp_name = st.text_input("Enter Company Name", key="comp_name",placeholder="enter name of the property developer")
        town_area = st.number_input("Enter Total TownShip Area in Acres", key="town_area", placeholder = "enter exact", min_value=1, max_value=20000, value="min")
    with col2:
        sub_Area = st.selectbox(
            "Select Sub-Area, (Select \"Other\" if not listed)*",
            sub_area_list,
            placeholder="Select Sub-Area..",
            index=0
            )    
        prop_area = st.text_input("Enter Property Area in Sq.ft.*", key="prop_area",placeholder="enter exact or range")
        town_name = st.text_input("Enter TownShip Name/ Society Name", key="town_name",placeholder="enter name of the property")
    
    
    col1, col2 = st.columns(2)

    with col1:
        clb_house = st.radio("Has Club House or recreational facility?", options, index=1, horizontal=True, key="clb_house")
        edu = st.radio("Has School / University in Township?", options, index=1, horizontal=True, key="edu")
        hosp = st.radio("Has Hospital in TownShip?", options, index=1, horizontal=True, key="hosp")
        mall = st.radio("Has Mall in TownShip?", options, index=1, horizontal=True, key="mall")
    with col2:
        park = st.radio("Has Park / Jogging track?", options, index=1, horizontal=True, key="park")
        pool = st.radio("Has Swimming Pool?", options, index=1, horizontal=True, key="pool")
        gym = st.radio("Has Gym?", options, index=1, horizontal=True, key="gym")

    data = [sub_Area, prop_type, prop_area, comp_name, town_name, town_area, options.index(clb_house), 
            options.index(edu), options.index(hosp), options.index(mall), options.index(park), options.index(pool), options.index(gym)]

    input_df = pd.DataFrame([data], columns=columns)

    submit_status = st.form_submit_button("Get Estimated Price")

    if submit_status:
        
        if input_df.isnull().values.any():
            st.warning('All Fields must be given', icon="⚠️")
        else:
            print("finding price")
            pred_obj = PredictionPipeline()
            price = pred_obj.predict(input_df)
            
            st.metric(label="Estimate", value=("Rs."+ str(round(price * 10, 3))+" Lakhs"))