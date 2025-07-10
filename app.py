import streamlit as st
import pandas as pd
import os
from PIL import Image
from pipeline import process_image

st.set_page_config(page_title="Fill Level Visualizer", layout="wide")
st.title(" Satellite Tank Fill Level Visualizer")

# Sidebar
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a satellite image", type=["jpg", "png", "jpeg"])
    run_btn = st.button("Run Analysis")

# Main
if uploaded_file:
    # Save uploaded image temporarily
    os.makedirs("images", exist_ok=True)
    image_path = os.path.join("images", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Original Satellite Image", use_column_width=True)

    if run_btn:
        with st.spinner("Running detection and fill estimation..."):
            df = process_image(image_path, model_path="weights/best.pt")

        st.success(" Processing complete!")

        # Show final image with boxes
        st.subheader("Detected Tanks")
        st.image("output/image_with_boxes.jpg", caption="Tanks with bounding boxes", use_column_width=True)

        # Table of fill percentages
        st.subheader("Estimated Fill Levels")
        st.dataframe(df)

        # Download button
        st.download_button(
            label=" Download CSV",
            data=df.to_csv(index=False),
            file_name="fill_estimates.csv",
            mime="text/csv"
        )

        # Intermediate visual inspection
        st.subheader("Inspect Individual Tanks")
        selected_id = st.number_input("Select tank ID", min_value=0, max_value=len(df)-1, step=1)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(f"output/intermediate/tank_{selected_id}_crop.png", caption="Tank Crop")
        with col2:
            st.image(f"output/intermediate/tank_{selected_id}_mask.png", caption="Circular Mask")
        with col3:
            st.image(f"output/intermediate/tank_{selected_id}_shadow.png", caption="Shadow Mask")

else:
    st.info("Upload a satellite image from the sidebar to begin.")
