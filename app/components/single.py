import streamlit as st
import pandas as pd

# You will plug this in later
# from src.predict_pipeline import predict_pump_status


def render_single_prediction():

    st.header("🔍 Single Pump Prediction")

    st.markdown(
        "Enter pump details below to predict its operational status."
    )

    # --------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------

    with st.form("single_prediction_form"):

        st.subheader("📍 Location Details")

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                "Latitude",
                value=0.0,
                format="%.6f"
            )

            gps_height = st.number_input(
                "GPS Height",
                min_value=-100,
                max_value=3000,
                value=500
            )

        with col2:
            longitude = st.number_input(
                "Longitude",
                value=0.0,
                format="%.6f"
            )

            lga = st.text_input(
                "LGA (Local Government Area)",
                value="Unknown"
            )

        st.divider()

        st.subheader("💧 Pump Details")

        col3, col4 = st.columns(2)

        with col3:
            construction_year = st.number_input(
                "Construction Year",
                min_value=1960,
                max_value=2026,
                value=2005
            )
            date_recorded = st.date_input(
                "Date Recorded"
            )
            population = st.number_input(
                "Population",
                min_value=0,
                value=1000
            )

            quantity = st.selectbox(
                "Water Quantity",
                [
                    "dry",
                    "insufficient",
                    "seasonal",
                    "enough"
                ]
            )

        with col4:
            basin = st.selectbox(
                "Basin",
                [
                    "Lake Victoria",
                    "Rufiji",
                    "Pangani",
                    "Internal",
                    "Lake Tanganyika"
                ]
            )

            extraction_type = st.selectbox(
                "Extraction Type",
                [
                    "gravity",
                    "handpump",
                    "motorpump",
                    "other"
                ]
            )

            waterpoint_type = st.selectbox(
                "Waterpoint Type",
                [
                    "communal standpipe",
                    "hand pump",
                    "cattle trough",
                    "other"
                ]
            )
            

        st.divider()

        st.subheader("🏢 Management Details")

        col5, col6 = st.columns(2)

        with col5:
            installer = st.text_input(
                "Installer",
                value="Unknown"
            )

        with col6:
            funder = st.text_input(
                "Funder",
                value="Unknown"
            )

        submitted = st.form_submit_button("🚀 Predict Pump Status")

    # --------------------------------------------------
    # ON SUBMIT
    # --------------------------------------------------

    if submitted:
        date_recorded = pd.to_datetime(date_recorded)  # Ensure date is in correct format
        input_data = pd.DataFrame([{
            "latitude": latitude,
            "longitude": longitude,
            "gps_height": gps_height,
            "lga": lga,
            "construction_year": construction_year,
            "population": population,
            "quantity": quantity,
            "basin": basin,
            "extraction_type": extraction_type,
            "waterpoint_type": waterpoint_type,
            "installer": installer,
            "funder": funder,
            "date_recorded": date_recorded
        }])

        st.subheader("📦 Input Summary")
        st.dataframe(input_data)

        # --------------------------------------------------
        # PREDICTION PLACEHOLDER (HOOK LATER)
        # --------------------------------------------------

        st.info("Model will be connected in next step.")

        # Example later:
        # prediction = predict_pump_status(input_data)
        # st.success(f"Prediction: {prediction}")