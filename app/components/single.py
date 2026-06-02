import streamlit as st
import pandas as pd

from src.predict_pipeline import predict_pump_status


def render_single_prediction():

    st.header("🔍 Single Pump Prediction")

    st.markdown(
        "Enter pump details below to predict the operational status of a water pump."
    )

    # --------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------

    with st.form("single_prediction_form"):

        # ==================================================
        # LOCATION DETAILS
        # ==================================================

        st.subheader("📍 Location Details")

        col1, col2 = st.columns(2)

        with col1:

            latitude = st.number_input(
                "Latitude",
                value=-6.0,
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
                value=35.0,
                format="%.6f"
            )

            lga = st.text_input(
                "LGA (Local Government Area)",
                value="Unknown"
            )

        st.divider()

        # ==================================================
        # PUMP DETAILS
        # ==================================================

        st.subheader("💧 Pump Details")

        col3, col4 = st.columns(2)

        with col3:

            amount_tsh = st.number_input(
                "Amount TSH",
                min_value=0.0,
                value=0.0
            )

            population = st.number_input(
                "Population",
                min_value=0,
                value=1000
            )

            construction_year = st.number_input(
                "Construction Year",
                min_value=1960,
                max_value=2026,
                value=2005
            )

            date_recorded = st.date_input(
                "Date Recorded"
            )

        with col4:

            basin = st.selectbox(
                "Basin",
                [
                    "Internal",
                    "Lake Nyasa",
                    "Lake Rukwa",
                    "Lake Tanganyika",
                    "Lake Victoria",
                    "Pangani",
                    "Rufiji",
                    "Ruvuma / Southern Coast",
                    "Wami / Ruvu"
                ]
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

            extraction_type = st.selectbox(
                "Extraction Type",
                [
                    "afridev",
                    "gravity",
                    "india mark ii",
                    "ksb",
                    "mono",
                    "nira/tanira",
                    "other",
                    "other - rope pump",
                    "others",
                    "submersible",
                    "swn 80"
                ]
            )

            waterpoint_type = st.selectbox(
                "Waterpoint Type",
                [
                    "cattle trough",
                    "communal standpipe",
                    "communal standpipe multiple",
                    "dam",
                    "hand pump",
                    "improved spring",
                    "other"
                ]
            )

        st.divider()

        # ==================================================
        # MANAGEMENT & FUNDING
        # ==================================================

        st.subheader("🏢 Management & Funding")

        col5, col6 = st.columns(2)

        with col5:

            installer = st.text_input(
                "Installer",
                value="Unknown"
            )

            management = st.text_input(
                "Management",
                value="other"
            )

            management_group = st.selectbox(
                "Management Group",
                [
                    "commercial",
                    "other",
                    "parastatal",
                    "user-group"
                ]
            )

            public_meeting = st.selectbox(
                "Public Meeting",
                [True, False]
            )

        with col6:

            funder = st.text_input(
                "Funder",
                value="Unknown"
            )

            scheme_management = st.text_input(
                "Scheme Management",
                value="other"
            )

            permit = st.selectbox(
                "Permit",
                [True, False]
            )

        # ==================================================
        # ADVANCED DETAILS
        # ==================================================

        with st.expander("⚙️ Advanced Water System Details"):

            col7, col8 = st.columns(2)

            with col7:

                payment = st.selectbox(
                    "Payment",
                    [
                        "never pay",
                        "other",
                        "pay annually",
                        "pay monthly",
                        "pay per bucket",
                        "pay when scheme fails"
                    ]
                )

                quality_group = st.selectbox(
                    "Quality Group",
                    [
                        "good",
                        "colored",
                        "fluoride",
                        "milky",
                        "salty"
                    ]
                )

            with col8:

                source_type = st.selectbox(
                    "Source Type",
                    [
                        "borehole",
                        "dam",
                        "other",
                        "rainwater harvesting",
                        "river/lake",
                        "shallow well",
                        "spring"
                    ]
                )

                source_class = st.selectbox(
                    "Source Class",
                    [
                        "groundwater",
                        "surface"
                    ]
                )

        st.divider()

        submitted = st.form_submit_button(
            "🚀 Predict Pump Status"
        )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if submitted:

        input_data = pd.DataFrame([{
            "amount_tsh": amount_tsh,
            "date_recorded": pd.to_datetime(date_recorded),
            "funder": funder,
            "gps_height": gps_height,
            "installer": installer,
            "longitude": longitude,
            "latitude": latitude,
            "basin": basin,
            "lga": lga,
            "population": population,
            "public_meeting": public_meeting,
            "scheme_management": scheme_management,
            "permit": permit,
            "construction_year": construction_year,
            "extraction_type": extraction_type,
            "management": management,
            "management_group": management_group,
            "payment": payment,
            "quality_group": quality_group,
            "quantity": quantity,
            "source_type": source_type,
            "source_class": source_class,
            "waterpoint_type": waterpoint_type
        }])

        st.subheader("📦 Input Summary")
        st.dataframe(input_data)

        try:

            with st.spinner("Generating prediction..."):

                prediction = predict_pump_status(
                    input_data
                )

            label_map = {
                0: "❌ Non Functional",
                1: "⚠️ Functional Needs Repair",
                2: "✅ Functional"
            }

            result = label_map.get(
                prediction,
                str(prediction)
            )

            st.success(
                f"Prediction: {result}"
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {str(e)}"
            )