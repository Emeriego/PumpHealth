import streamlit as st
import pandas as pd

from src.predict_pipeline import predict_pump_status, predict_batch


def render_batch_prediction():

    st.header("📊 Batch Prediction (CSV Upload)")

    st.markdown(
        """
        Upload a CSV file containing multiple pump records.
        The system will generate predictions for each row.
        """
    )

    # --------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("Please upload a CSV file to continue.")
        return

    try:
        df = pd.read_csv(uploaded_file)

        st.success(
            f"File loaded: {df.shape[0]} rows × {df.shape[1]} columns"
        )

        st.subheader("🔎 Data Preview")
        st.dataframe(df.head())

        # --------------------------------------------------
        # REQUIRED COLUMNS CHECK
        # --------------------------------------------------

        required_cols = [
            "amount_tsh",
            "date_recorded",
            "funder",
            "gps_height",
            "installer",
            "longitude",
            "latitude",
            "basin",
            "lga",
            "population",
            "public_meeting",
            "scheme_management",
            "permit",
            "construction_year",
            "extraction_type",
            "management",
            "management_group",
            "payment",
            "quality_group",
            "quantity",
            "source_type",
            "source_class",
            "waterpoint_type"
        ]

        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns:\n{missing_cols}")
            return

        st.success("All required columns are present.")

        # --------------------------------------------------
        # RUN PREDICTION
        # --------------------------------------------------

        if st.button("🚀 Run Batch Prediction"):

            with st.spinner("Generating predictions..."):

                results = predict_batch(df)

            st.success("Prediction completed!")

            st.subheader("📈 Results Preview")
            st.dataframe(results.head())

            # --------------------------------------------------
            # DOWNLOAD RESULTS
            # --------------------------------------------------

            csv = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions",
                data=csv,
                file_name="pump_predictions.csv",
                mime="text/csv"
            )

            # --------------------------------------------------
            # OPTIONAL SUMMARY
            # --------------------------------------------------

            if "prediction" in results.columns:

                st.subheader("📊 Prediction Summary")

                st.write(
                    results["prediction"].value_counts()
                )

    except Exception as e:

        st.error(f"Error processing file: {str(e)}")