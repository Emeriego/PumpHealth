import streamlit as st
import pandas as pd

# later you will plug this in
# from src.predict_pipeline import predict_batch


def render_batch_prediction():

    st.header("📊 Batch Prediction (CSV Upload)")

    st.markdown(
        """
        Upload a CSV file containing multiple pump records.
        The system will return predictions for all rows.
        """
    )

    # --------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            st.success(
                f"File loaded successfully with {df.shape[0]} rows "
                f"and {df.shape[1]} columns."
            )

            st.subheader("🔎 Data Preview")
            st.dataframe(df.head())

            # --------------------------------------------------
            # BASIC VALIDATION (IMPORTANT)
            # --------------------------------------------------

            required_cols = [
                "latitude",
                "longitude",
                "gps_height",
                "construction_year",
                "population",
                "quantity",
                "basin",
                "extraction_type",
                "waterpoint_type",
                "installer",
                "funder",
                "lga"
            ]

            missing_cols = [
                col for col in required_cols if col not in df.columns
            ]

            if missing_cols:
                st.error(
                    f"Missing required columns: {missing_cols}"
                )
                return

            st.success("All required columns are present.")

            # --------------------------------------------------
            # PREDICTION PLACEHOLDER
            # --------------------------------------------------

            st.info("Model integration will be added next step.")

            # Later:
            # results = predict_batch(df)
            # st.dataframe(results)

            # csv download example:
            # csv = results.to_csv(index=False).encode("utf-8")
            # st.download_button(
            #     "Download Predictions",
            #     csv,
            #     "predictions.csv",
            #     "text/csv"
            # )

        except Exception as e:
            st.error(f"Error reading file: {e}")

    else:
        st.info("Please upload a CSV file to continue.")