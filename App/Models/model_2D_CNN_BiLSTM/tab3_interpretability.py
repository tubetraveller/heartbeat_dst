import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
from Models.model_2D_CNN_BiLSTM.tab2_model_quality import load_cnn_bilstm_model, load_features, load_xgb_model
from io import BytesIO
import lime.lime_tabular
from streamlit import components

@st.cache_data
# Function to generate and display the SHAP plot
def shap_plot_streamlit():
    """
    Generates and displays a SHAP plot in the Local Interpretability section.
    """
    # Display the pre-generated LIME image
    st.image("./Models/2D_CNN_BiLSTM/shap_plot.svg", caption="LIME Explanation", width=1200, use_container_width=False)
    

@st.cache_data
# Function to generate and display the LIME explanation
def lime_plot_streamlit():
    """
    Displays a pre-generated LIME explanation as an image in Streamlit.
    """
    # Display the pre-generated LIME image
    st.image("./Models/2D_CNN_BiLSTM/lime.svg", caption="LIME Explanation", width=1100, use_container_width=False)


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
    with tab2:  # Generate and display LIME explanation
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
        lime_plot_streamlit()

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
        col1, col2 = st.columns([1, 1])  # Two equal-width columns

        with col1:
            global_shap_plot_with_image()

        with col2:
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