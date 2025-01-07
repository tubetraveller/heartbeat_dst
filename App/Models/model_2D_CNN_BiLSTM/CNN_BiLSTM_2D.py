import streamlit as st
from Models.model_2D_CNN_BiLSTM.tab1_model_overview import model_overview
from Models.model_2D_CNN_BiLSTM.tab2_model_quality import model_quality
from Models.model_2D_CNN_BiLSTM.tab3_interpretability import interpretability
from Models.model_2D_CNN_BiLSTM.tab4_output_classes import output_classes


def show_2d_cnn_bilstm():
    st.header("2D-CNN-BiLSTM Model with XGBoost for Heartbeat Classification")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Model Overview", "Model Quality", "Interpretability", "Output Classes"])

    # Call corresponding functions for each tab
    with tab1:
        model_overview()
    with tab2:
        model_quality()
    with tab3:
        interpretability()
    with tab4:
        output_classes()
