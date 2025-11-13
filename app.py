import streamlit as st
import pandas as pd
import pickle
import os
import io

# Increase upload limit (in MB)
st.set_option("server.maxUploadSize", 1024)

st.title("Customer Churn Prediction App")
st.write("Upload a dataset and get churn predictions instantly.")

# === Step 1: Load the trained model ===
model_path = "model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found. Please upload or train the model first.")
else:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
        st.stop()

# === Step 2: Upload test data ===
uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.dataframe(data.head())

        if "customer_id" not in data.columns:
            st.warning("No 'customer_id' column found — adding placeholder IDs.")
            data["customer_id"] = range(1, len(data) + 1)

        # === Step 3: Make predictions ===
        preds = model.predict(data.drop("customer_id", axis=1, errors="ignore"))
        data["churn"] = preds

        # === Step 4: Display and download results ===
        st.success("Predictions complete!")
        st.dataframe(data[["customer_id", "churn"]].head())

        csv = data[["customer_id", "churn"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
