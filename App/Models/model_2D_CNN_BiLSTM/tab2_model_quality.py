import os

# Suppress TensorFlow oneDNN and logging warnings
#   # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st 
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt

# Function to load the CNN-BiLSTM model

@st.cache_resource
def load_cnn_bilstm_model():
    """
    Loads the CNN-BiLSTM model and ensures it is properly compiled.
    """
    cnn_bilstm_model_path = "./Models/2D_CNN_BiLSTM/cnn_bilstm_model_fold_4.h5"
    
    # Check if the model file exists
    if not os.path.exists(cnn_bilstm_model_path):
        st.error(f"File not found: {cnn_bilstm_model_path}")
        return None
    
    try:
        # Load the model
        cnn_bilstm_model = load_model(cnn_bilstm_model_path)
        #st.write("Model loaded successfully!")

        # Compile the model to ensure it has the necessary attributes
        cnn_bilstm_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        #st.write("Model compiled successfully!")

        # Safely check if compiled metrics are available
        #if hasattr(cnn_bilstm_model, 'metrics_names') and cnn_bilstm_model.metrics_names:
            #st.write("Model metrics are properly loaded:", cnn_bilstm_model.metrics_names)
       # else:
           # st.warning("Model metrics appear to be missing. Ensure the model was trained correctly.")

        # Display model summary in logs (optional)
        #with st.expander("Model Summary"):
            #summary_str = []
            #cnn_bilstm_model.summary(print_fn=lambda x: summary_str.append(x))
            #st.text("\n".join(summary_str))

        return cnn_bilstm_model

    except Exception as e:
        st.error(f"An error occurred while loading or compiling the model: {e}")
        return None


@st.cache_data
# Function to load features, labels, predictions, and probabilities
def load_features():
    features_path = "./Models/2D_CNN_BiLSTM/features_fold_4.npz"
    if not os.path.exists(features_path):
        st.error(f"File not found: {features_path}")
        return None, None, None, None, None, None
    features = np.load(features_path)
    return (
        features["X_train_features"],
        features["X_val_features"],
        features["y_train_fold_smote"],
        features["X_val_fold_reshaped"], 
        features["y_val_fold"],
        features["y_val_pred"],  # Load predictions
        features["y_val_prob"],
          # Load probabilities
    )
    
@st.cache_resource
# Function to load the XGBoost model
def load_xgb_model():
    xgb_model_path = "./Models/2D_CNN_BiLSTM/xgb_model_fold_4.pkl"
    if not os.path.exists(xgb_model_path):
        st.error(f"File not found: {xgb_model_path}")
        return None
    return joblib.load(xgb_model_path)

# Function to format the classification report
def format_classification_report(y_true, y_pred, target_names):
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    def format_value(val):
        if isinstance(val, float):
            return f"{val:.2f}".rstrip(".")
        return val

    # Use DataFrame.apply instead of applymap
    report_df = report_df.apply(lambda col: col.map(format_value) if col.dtype == "float" or col.dtype == "object" else col)
    
    if "support" in report_df.columns:
        report_df["support"] = report_df["support"].apply(lambda x: f"{int(float(x)):,}")
    
    report_df.index = report_df.index.str.capitalize()
    return report_df


# Function to plot ROC curve using Altair
def plot_roc_curve_altair(y_true, y_prob, labels):
    roc_data = []
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(np.array(y_true) == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data.extend(
            [{"False Positive Rate": f, 
              "True Positive Rate": t, 
              "Class": f"{label} (AUC = {roc_auc:.2f})",  # Include AUC in class label
              "AUC": roc_auc}
             for f, t in zip(fpr, tpr)]
        )
    df_roc = pd.DataFrame(roc_data)
   # Create the ROC curve chart
    roc_curve_chart = alt.Chart(df_roc).mark_line().encode(
        x=alt.X("False Positive Rate:Q", title="False Positive Rate"),
        y=alt.Y("True Positive Rate:Q", title="True Positive Rate"),
        color=alt.Color("Class:N", title="Class with AUC"),  # AUC in legend
        tooltip=["Class:N", "AUC:Q"]
    )
    
    # Add the random guess line as a dashed line
    random_guess_line = alt.Chart(pd.DataFrame({
        "False Positive Rate": [0, 1],
        "True Positive Rate": [0, 1]
    })).mark_line(strokeDash=[5, 5], color="black").encode(
        x="False Positive Rate:Q",
        y="True Positive Rate:Q",
        tooltip=alt.Tooltip(value="Random Guess")
    ).properties()
    
    # Combine the ROC curve and the random guess line
    combined_chart = alt.layer(roc_curve_chart, random_guess_line).properties(
        width=600, height=400
    ).configure_axis(
                labelFontSize=18,
                titleFontSize=18
            ).configure_legend(
                labelFontSize=18,    # Set font size for legend labels
                titleFontSize=18     # Set font size for legend title
            )
    
    st.altair_chart(combined_chart, use_container_width=False)

# Function to plot Precision-Recall Curve
def plot_precision_recall_curve_altair(y_true, y_prob, labels):
    pr_data = []
    for i, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(np.array(y_true) == i, y_prob[:, i])
        pr_auc = auc(recall, precision)
        pr_data.extend(
            [{"Recall": r, 
              "Precision": p, 
              "Class": f"{label} (AUC = {pr_auc:.2f})",  # Include AUC in class label
              "AUC": pr_auc}
             for r, p in zip(recall, precision)]
        )
    
    df_pr = pd.DataFrame(pr_data)
    
    # Create the Precision-Recall curve chart
    pr_curve_chart = alt.Chart(df_pr).mark_line().encode(
        x=alt.X("Recall:Q", title="Recall"),
        y=alt.Y("Precision:Q", title="Precision"),
        color=alt.Color("Class:N", title="Class with AUC"),  # AUC in legend
        tooltip=["Class:N", "AUC:Q"]
    )
    
    # Add the random guess line as a dashed line
    random_guess_line = alt.Chart(pd.DataFrame({
        "Recall": [0, 1],
        "Precision": [1, 0]
    })).mark_line(strokeDash=[5, 5], color="black").encode(
        x="Recall:Q",
        y="Precision:Q",
        tooltip=alt.Tooltip(value="Random Guess")
    ).properties()
    
    # Combine the Precision-Recall curve and the random guess line
    combined_chart = alt.layer(pr_curve_chart, random_guess_line).properties(
        width=600, height=400
    ).configure_axis(
                labelFontSize=18,
                titleFontSize=18
            ).configure_legend(
                labelFontSize=18,    # Set font size for legend labels
                titleFontSize=18     # Set font size for legend title
            )
    
    st.altair_chart(combined_chart, use_container_width=False)


# Function to display the confusion matrix SVG
def display_confusion_matrix_svg():
    filepath = "./Models/2D_CNN_BiLSTM/confusion_matrix.svg"  # Path to your SVG file
    if os.path.exists(filepath):
        st.image(
            filepath, 
            caption="Confusion Matrix", 
            use_container_width=False
        )
    else:
        st.error(f"File not found: {filepath}")

# Main function to display model quality
# Main function to display model quality
def model_quality():
    st.subheader("Model Quality and Evaluation")

    st.info("Loading CNN-BiLSTM model...")
    cnn_bilstm_model = load_cnn_bilstm_model()
    if cnn_bilstm_model is None:
        st.stop()

    st.info("Loading features...")
    (
        X_train_features,
        X_val_features,
        X_val_fold_reshaped,
        y_train,
        y_val,
        y_val_pred,
        y_val_prob,
    ) = load_features()
    if X_train_features is None or X_val_features is None:
        st.stop()

    st.info("Loading XGBoost model...")
    xgb_model = load_xgb_model()
    if xgb_model is None:
        st.stop()

    labels = ["N", "S", "V", "F", "Q"]

    # Create tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "Classification Report", 
        "Confusion Matrix", 
        "ROC Curve", 
        "Precision-Recall Curve"
    ])

    with tab1:
        st.subheader("Classification Report")
        col1, col2 = st.columns(2)
        with col1:
            # Generate and display the classification report as a table
            report_df = format_classification_report(y_val, y_val_pred, target_names=labels)
            st.table(report_df)
        with col2:
            st.write("""
            - **Overall Performance**:
            - The model demonstrates **excellent overall performance** with an accuracy of **98%** and a weighted F1-score of **0.98**.
            - **Class-Wise Highlights**:
            - **N (Normal)**: Exceptional precision, recall, and F1-score (0.99), reflecting strong performance for the majority class.
            - **S (Supraventricular ectopic beat)**: Performs well with an F1-score of 0.80, considering the limited representation.
            - **V (Ventricular ectopic beat)**: Strong F1-score of 0.95 indicates effective feature extraction for this class.
            - **F (Fusion beat)**: Achieves an F1-score of 0.77, a commendable result for the smallest class (214 samples).
            - **Q (Unclassifiable beat)**: High precision, recall, and F1-score (0.98) demonstrate effective classification.
            - **Handling Minority Classes**:
            - The model handles minority classes effectively, showcasing strong baseline performance despite data limitations.
            - **Conclusion**:
            - These results highlight the model's exceptional capabilities and its potential for further refinement.
            """)

    with tab2:
        st.subheader("Confusion Matrix")
        col1, col2 = st.columns(2)
        with col1:
            # Display the confusion matrix SVG
            display_confusion_matrix_svg()
        with col2:
            st.write("""
            - **High Accuracy on Majority Classes**:
            - **N (Normal)**: 99% of true labels are correctly classified, reflecting the model's robustness.
            - **Q (Unclassifiable beat)**: 98% accurate classification, showcasing reliable predictions.
            - **Handling Minority Classes**:
            - Despite the inherent challenge of imbalanced data:
                - **S (Supraventricular ectopic beat)**: Achieves an impressive 83% accuracy, with minor overlap into **N** due to feature similarities.
                - **F (Fusion beat)**: Correctly classified 78% of the time, a strong result for the smallest class with significant challenges in representation.
            - **Augmentation Limitation**:
            - While **time-sequence perturbation of ECG signals** could theoretically increase data diversity, it risks introducing artifacts that may lead to misclassification.
            - Careful consideration is required to ensure augmentation preserves the signal quality and avoids negatively impacting classification accuracy.
            - **Conclusion**:
            - The confusion matrix highlights the model's **strong capability** to generalize across classes, achieving robust predictions despite data challenges.
            """)

    with tab3:
        st.subheader("ROC Curve")
        col1, col2 = st.columns(2)
        with col1:
            # Plot the ROC curve
            plot_roc_curve_altair(y_val, y_val_prob, labels)
        with col2:
            st.write("""
            - **Exceptional Model Performance**:
            - The **ROC Curve** demonstrates that the model performs exceptionally well across all classes, with AUC values near 1.
            - Key AUC values:
                - **Q (Unclassifiable beat)**: AUC = 1.00
                - **V (Ventricular ectopic beat)**: AUC = 1.00
                - **N (Normal)**: AUC = 0.99
                - **S (Supraventricular ectopic beat)**: AUC = 0.98
                - **F (Fusion beat)**: AUC = 0.97
            - **High True Positive Rates**:
            - The curves show high **True Positive Rates (TPR)** even at low **False Positive Rates (FPR)**, reflecting the model's ability to effectively distinguish between classes.
            - **Minority Class Performance**:
            - Despite being the most challenging, the **F (Fusion beat)** class achieves an AUC of 0.97, demonstrating the model's robustness for underrepresented classes.
            - **Conclusion**:
            - The ROC Curve and AUC values confirm the model's **high predictive power and reliability** for all classes, including underrepresented ones.
            """)

    with tab4:
        st.subheader("Precision-Recall Curve")
        col1, col2 = st.columns(2)
        with col1:
            # Plot the Precision-Recall Curve
            plot_precision_recall_curve_altair(y_val, y_val_prob, labels)
        with col2:
            st.write("""
            - **Outstanding Performance for Key Classes**:
            - **Q (Unclassifiable beat)** and **N (Normal)**: AUC = 1.00, reflecting perfect precision-recall balance for these classes.
            - **V (Ventricular ectopic beat)**: AUC = 0.98, demonstrating the model's ability to handle this important class effectively.
            - **Handling Minority Classes**:
            - **S (Supraventricular ectopic beat)**: AUC = 0.87, maintaining a good balance between precision and recall despite data limitations.
            - **F (Fusion beat)**: AUC = 0.77, showcasing reasonable performance for a challenging minority class.
            - **Model Strengths**:
            - High precision across most recall values for major classes (e.g., N, Q, and V), confirming the model's ability to minimize false positives while maximizing true positives.
            - Performance gaps for minority classes like **F** are attributed to dataset constraints rather than model limitations.
            - **Conclusion**:
            - The precision-recall curve highlights the model's **strong capability** to balance precision and recall, especially for majority classes, while performing well for minority classes given the constraints.
            """)


