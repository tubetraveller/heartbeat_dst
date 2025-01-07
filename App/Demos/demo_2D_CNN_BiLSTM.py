import streamlit as st
import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# Import necessary functions
from Models.model_2D_CNN_BiLSTM.tab2_model_quality import load_cnn_bilstm_model, load_features

# Load the full MIT-BIH train dataset
@st.cache_data
def load_full_mitbih_train():
    mitbih_train = pd.read_csv('./Data/mitbih_train.csv', header=None)  # Path to full dataset
    return mitbih_train

def show_demo_2d_cnn_bilstm():
    """
    Displays the SHAP plot for a selected row from MIT-BIH Train dataset with expert label comparison.
    """
    st.header("SHAP Demo: MIT-BIH Train Dataset")

    # Step 1: Load the Model
    st.subheader("1. Load Model")
    with st.spinner("Loading the 2D CNN-BiLSTM model..."):
        model = load_cnn_bilstm_model()  # Load the model
    st.success("Model loaded successfully!")

    # Step 2: Load Dataset and Select Row
    st.subheader("2. Select Row for Class Determination")
    _, _, _, X_val_fold_reshaped, _, _, _ = load_features()  # Assuming this function returns the reshaped validation data

    # Load full MIT-BIH train dataset
    mitbih_train = load_full_mitbih_train()

    # Select a row index
    selected_index = st.number_input(
        "Enter the row index (0 to {}):".format(len(X_val_fold_reshaped) - 1),
        min_value=0,
        max_value=len(X_val_fold_reshaped) - 1,
        value=27031,  # Default to the example row index
        step=1
    )
    selected_sample = X_val_fold_reshaped[selected_index:selected_index + 1]

    # Map to original MIT-BIH row
    original_index = int(selected_index * 3)  # Map to the corresponding row in the original dataset
    expert_label_numeric = mitbih_train.iloc[original_index, -1]  # Get the label from the last column
    all_classes = ["N Class", "S Class", "V Class", "F Class", "Q Class"]
    expert_label = all_classes[int(expert_label_numeric)]

    # Display selected row info and expert label
    st.write(f"Selected Row Index (Subset): {selected_index}")
    #st.write(f"Mapped to Original Row (Full Dataset): {original_index}")
    st.markdown(
        f"""
        <p style="background-color: #fff0f8; padding: 10px; border-radius: 5px; font-size: 16px;">
            Expert-determined Label: <b>{expert_label}</b>
        </p>
        """,
        unsafe_allow_html=True
    )


    #st.write(f"Sample shape: {selected_sample.shape}")

    # Step 3: Generate SHAP Plot
    st.subheader("3. Prediction of the Model")
    if st.button("Plot SHAP Explanation"):
        with st.spinner("Generating SHAP plot..."):
            # Load masker_shape from the saved file
            masker_shape_path = "./Models/2D_CNN_BiLSTM/masker_shape.npz"
            try:
                masker_shape_data = np.load(masker_shape_path)
                masker_shape = masker_shape_data["masker_shape"]
                masker_blur = shap.maskers.Image("blur(128,128)", masker_shape)
            except FileNotFoundError:
                st.error(f"File not found: {masker_shape_path}")
                return

            # Define SHAP explainer
            def predict(images):
                return model.predict(images)

            explainer = shap.Explainer(predict, masker_blur)

            # Generate SHAP values for the selected sample
            shap_values = explainer(
                selected_sample,
                max_evals=1000,
                batch_size=80,
                outputs=shap.Explanation.argsort.flip[:6],  # Generate values for 6 outputs
            )

            # Post-process SHAP values for visualization
            shap_values.data = selected_sample[0]
            shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

            # Titles for each image based on prediction
            predicted_probs = model.predict(selected_sample)
            predicted_class = np.argmax(predicted_probs)
            titles = [f"Predicted Class: {all_classes[predicted_class]}"]
            for class_name in all_classes:
                if class_name != all_classes[predicted_class]:
                    titles.append(class_name)

            # Use BytesIO to capture the SHAP plot
            buf = BytesIO()
            fig, ax = plt.subplots(figsize=(14, 10))
            shap.image_plot(
                shap_values=shap_values.values,
                pixel_values=shap_values.data,
                labels=titles,
                show=False  # Do not call plt.show()
            )
            plt.savefig(buf, format="png", dpi=600, bbox_inches="tight")  # Save the figure to the buffer
            buf.seek(0)

            # Display the SHAP plot in Streamlit
            st.image(buf, caption=f"SHAP Explanation for Row {selected_index}", use_container_width=False)
            buf.close()  # Close the buffer to free memory
