import streamlit as st

from model_helper import predict
st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload your file", type= ["jpeg", "png"])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption= "Upload File", use_column_width=True)
        prediction = predict(image_path)
        st.info(f"Predicted Class: {prediction}")