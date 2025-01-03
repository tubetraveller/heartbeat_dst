
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
@st.cache_resource
# Function to load the CNN-BiLSTM model
def load_cnn_bilstm_model():
    cnn_bilstm_model_path = "./Models/2D_CNN_BiLSTM/cnn_bilstm_model_fold_4.h5"
    if not os.path.exists(cnn_bilstm_model_path):
        st.error(f"File not found: {cnn_bilstm_model_path}")
        return None
    cnn_bilstm_model = load_model(cnn_bilstm_model_path)
    cnn_bilstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn_bilstm_model

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
        # Generate report using loaded predictions
        report_df = format_classification_report(y_val, y_val_pred, target_names=labels)
        st.table(report_df)

    with tab2:
        st.subheader("Confusion Matrix")
        display_confusion_matrix_svg()

    with tab3:
        st.subheader("ROC Curve")
        plot_roc_curve_altair(y_val, y_val_prob, labels)

    with tab4:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve_altair(y_val, y_val_prob, labels)



