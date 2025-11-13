import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction App")
st.write("Upload your customer data CSV file to predict churn dynamically.")

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview:")
    st.write(df.head())

    # Load model
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
        st.stop()

    # Make predictions
    predictions = model.predict(df)
    df["Predicted_Churn"] = predictions

    # Show results
    st.subheader("🔮 Predictions:")
    st.write(df.head())

    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv")

    st.success("Prediction completed successfully")
