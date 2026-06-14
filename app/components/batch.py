import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.predict_pipeline import predict_batch


def render_batch_prediction():

    st.header("Batch Prediction (CSV Upload)")

    st.markdown(
        """
        Upload a CSV file containing multiple pump records.
        The system will generate predictions for each row.
        """
    )
    st.markdown("""
    <style>

    div.stButton > button {
        background-color: #2563eb;
        color: white;
        font-size: 22px;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        height: 60px;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
    }

    </style>
    """, unsafe_allow_html=True)
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

        st.subheader("Data Preview")
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

        if st.button("Run Batch Prediction"):

            with st.spinner("Generating predictions..."):

                results = predict_batch(df)

            st.success("Prediction completed!")

            st.subheader("Results Preview")
            st.write(f"Showing first 100 of {len(results):,} predictions")

            # st.dataframe(
            #     results.head(100),
            #     use_container_width=True
            # )
            # Map prediction codes to labels
            prediction_map = {
                0: "Functional",
                1: "Functional Needs Repair",
                2: "Non Functional"
            }

            # Create a copy so the original results remain unchanged
            display_results = results.copy()

            display_results["prediction"] = (
                display_results["prediction"]
                .map(prediction_map)
                .fillna(display_results["prediction"])
            )

            # Display the first 100 rows
            st.dataframe(
                display_results.head(100),
                use_container_width=True
            )
            # --------------------------------------------------
            # DOWNLOAD RESULTS
            # --------------------------------------------------

            csv = display_results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions",
                data=csv,
                file_name="pump_predictions.csv",
                mime="text/csv"
            )

            # --------------------------------------------------
            # SUMMARY
            # --------------------------------------------------

            if "prediction" in display_results.columns:

                st.subheader("Prediction Summary")

                summary = display_results["prediction"].value_counts()


                st.dataframe(
                    summary.rename_axis("Status")
                        .reset_index(name="Count"),
                    use_container_width=True
                )

                fig, ax = plt.subplots(figsize=(3, 3))

                ax.pie(
                    summary.values,
                    labels=summary.index,
                    autopct="%1.1f%%"
                )

                st.pyplot(fig)

    except Exception as e:

        st.error(f"Error processing file: {str(e)}")