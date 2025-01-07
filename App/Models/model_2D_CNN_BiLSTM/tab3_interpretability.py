import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
from Models.model_2D_CNN_BiLSTM.tab2_model_quality import load_cnn_bilstm_model, load_features
from io import BytesIO
import lime.lime_tabular
import streamlit.components.v1 as components

@st.cache_data
# Function to generate and display the SHAP plot
def shap_plot_streamlit():
    """
    Generates and displays a SHAP plot in the Local Interpretability section.
    """
    # Load features and model
    (
        X_train_features,
        X_val_features,
        y_train_fold_smote,
        X_val_fold_reshaped,
        y_val_fold,
        y_val_pred,
        y_val_prob,
    ) = load_features()  # Exclude masker_shape here

    model = load_cnn_bilstm_model()

    if model is None:
        st.error("Model not loaded. Please load the model in Tab 2.")
        return

    # Load masker_shape from the saved file
    masker_shape_path = "./Models/2D_CNN_BiLSTM/masker_shape.npz"
    try:
        masker_shape_data = np.load(masker_shape_path)
        masker_shape = masker_shape_data["masker_shape"]
    except FileNotFoundError:
        st.error(f"File not found: {masker_shape_path}")
        return

    # Initialize SHAP masker using masker_shape
    masker_blur = shap.maskers.Image("blur(128,128)", masker_shape)

    # Define SHAP explainer
    def predict(images):
        return model.predict(images)

    explainer = shap.Explainer(predict, masker_blur)

    # Select a sample for SHAP analysis
    X_val_sample = X_val_fold_reshaped[27031:27032]
    st.write(f"Sample shape for SHAP: {X_val_sample.shape}")

    # Predict the class for the sample
    predicted_probs = model.predict(X_val_sample)
    predicted_class = np.argmax(predicted_probs)

    # Titles for each image based on prediction
    all_classes = ["N Class", "S Class", "V Class", "F Class", "Q Class"]
    titles = [f"Predicted Class: {all_classes[predicted_class]}"]
    for class_name in all_classes:
        if class_name != all_classes[predicted_class]:
            titles.append(class_name)

    # Generate SHAP values
    shap_values = explainer(
        X_val_sample,
        max_evals=1000,
        batch_size=80,
        outputs=shap.Explanation.argsort.flip[:len(titles) + 1],
    )

    # Post-process SHAP values for visualization
    shap_values.data = X_val_sample[0]
    shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

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
    st.image(buf, caption="SHAP Explanation", use_container_width=False)
    buf.close()  # Close the buffer to free memory

@st.cache_data
# Function to generate and display the LIME explanation
def lime_plot_streamlit():
    """
    Displays the LIME explanation as an HTML file in Streamlit with controlled size and alignment.
    """
    html_path = "./Models/2D_CNN_BiLSTM/lime_plot.html"

    try:
        with open(html_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        # Inject custom CSS for size and alignment control
        custom_css = """
        <style>
            .custom-lime-container {
                max-width: 900px;  /* Adjust max width */
                max-height: 700px;  /* Adjust max height */
                margin: off;       /* Center alignment */
                overflow: on;     /* Add scrollbars if necessary */
                border: 0.5px solid #ddd;  /* Optional: Add border for better visualization */
                padding: 0.5px;      /* Optional: Add padding */
                box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1); /* Optional: Add shadow */
            }

            .container.body-content {
                width: 80%;        /* Full width of the content */
                padding: 0;         /* Remove padding */
                margin: 0;          /* Remove margin */
            }
        </style>
        """
        # Embed the LIME HTML within a custom div for styling
        html_with_css = f"""
        {custom_css}
        <div class="custom-lime-container">
            {html_content}
        </div>
        """

        # Render the styled HTML in Streamlit
        st.components.v1.html(html_with_css, height=1200, scrolling=True)  # Adjust height to fit content
    except FileNotFoundError:
        st.error(f"File not found: {html_path}")
    except Exception as e:
        st.error(f"An error occurred while displaying the LIME explanation: {e}")





# Local Interpretability Section
def local_interpretability():
    """
    Displays the Local Interpretability tab with SHAP and LIME sub-tabs.
    """
    tab1, tab2 = st.tabs(["SHAP", "LIME"])
    with tab1:
        shap_plot_streamlit()  # Generate and display SHAP plot
        st.write("""
- **Model Interpretability**:
  - The SHAP value visualizations provide insights into how the model makes predictions by identifying feature contributions for each class.
- **Class-Specific Explanations**:
  - **Predicted Class: F (Fusion beat)**: Highlights specific signal regions with both positive (red) and negative (blue) contributions that influence the prediction.
  - **N (Normal)** and **S (Supraventricular ectopic beat)**: SHAP values illustrate contributing areas, helping understand potential misclassification risks.
  - **V (Ventricular ectopic beat)** and **Q (Unclassifiable beat)**: Clear regions of importance reflect the model's focus on class-specific signal characteristics.
- **Positive Contribution**:
  - Positive SHAP values (red) indicate features pushing the model toward a specific class.
  - Negative SHAP values (blue) highlight features reducing the likelihood of a given class.
- **Model Strengths**:
  - The analysis shows that the model learns meaningful patterns and focuses on distinct ECG signal areas, aligning with domain knowledge.
- **Conclusion**:
  - SHAP-based interpretability validates the model's decision-making process and supports its ability to generalize effectively across classes.
""")
    with tab2:
        lime_plot_streamlit()  # Generate and display LIME explanation
        st.write("""
- **Model Prediction Confidence**:
  - The model confidently predicts the instance as **Class F (Fusion beat)** with a probability of **1.00**, indicating high certainty in its decision.
- **Feature Contribution Analysis**:
  - **Key Positive Contributors**:
    - **Feature 68 (2.57)**, **Feature 69 (2.68)**, and **Feature 70 (1.34)** strongly drive the prediction toward Class F.
  - **Key Negative Contributors**:
    - Features like **Feature 73 (-1.33)** and **Feature 72 (-1.11)** reduce the likelihood of other classes, reinforcing the decision for Class F.
- **Interpretability Insights**:
  - The LIME explanation highlights the specific features that the model relied upon for its prediction, ensuring transparency in decision-making.
- **Model Strengths**:
  - Pinpointing positive and negative feature contributions demonstrates interpretability of model and alignment with explainable AI principles.
- **Conclusion**:
  - LIME validates the reliability of the model's prediction by linking it to influential features, offering confidence in its performance for Class F.
""")

@st.cache_resource
# Function to load SHAP values
def load_shap_values():
    shap_values_path = "./Models/2D_CNN_BiLSTM/shap_values.npz"
    try:
        data = np.load(shap_values_path)
        shap_values_array = data["shap_values_array"]
        return shap_values_array
    except FileNotFoundError:
        st.error(f"SHAP values file not found: {shap_values_path}")
        return None

# Function to generate and display the SHAP summary plot
@st.cache_data
def global_shap_plot_with_image():
    """
    Displays the SHAP global plot as an SVG image in Streamlit.
    """
    st.image("./Models/2D_CNN_BiLSTM/global_shap_plot.svg", caption="SHAP Summary Plot: Global Feature Importance", width=550, use_container_width=False)


# Main Interpretability Tab
def interpretability():
    """
    Main Interpretability tab with Local and Global Interpretability sections.
    """
    st.subheader("Model Interpretability")

    tab1, tab2 = st.tabs(["Local Interpretability", "Global Interpretability"])
    with tab1:
        local_interpretability()  # Call Local Interpretability
    with tab2:
        
        global_shap_plot_with_image()
        st.write("""
- **Feature Value Insights**:
  - The color gradient (blue to red) represents feature values:
    - **Red (High values)**: Positive contribution to the model’s prediction.
    - **Blue (Low values)**: Negative impact or reduced contribution to the prediction.
- **Top Influential Features**:
  - **Feature 79**: The most impactful feature, with consistently high SHAP values, strongly influencing the model’s output.
  - **Feature 2** and **Feature 78**: Significant contributors, with a mix of positive (red) and negative (blue) impacts depending on their values.
- **Feature Contribution Patterns**:
  - **Feature 83** and **Feature 101**: Split between high and low contributions, crucial for distinguishing certain classes.
  - **Feature 77 and Feature 39**: Moderate but consistent impact, likely influencing edge cases or minority class predictions.
- **Global Impact on Predictions**:
  - **Feature 79** and **Feature 2** dominate globally, reliably influencing predictions across samples.
  - Their wide SHAP value distribution reflects sensitivity to subtle variations in data.
- **Class-Specific Behavior**:
  - Features like **Feature 79** and **Feature 78** likely drive predictions for dominant classes (**N** and **Q**).
  - Moderate contributors (e.g., **Feature 77**) assist in predicting minority classes such as **S** or **F**.
- **Conclusion**:
  - The SHAP summary plot highlights specific feature contributions, validating the model's decisions and aligning them with meaningful data patterns.
""")