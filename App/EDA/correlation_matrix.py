import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Function to compute correlation matrix and cache the result
@st.cache_data
def compute_correlation_matrix(df):
    return df.corr()

@st.cache_data
# Function to plot heatmap using Altair
def plot_heatmap(corr_matrix, title):
    # Transform correlation matrix to long-form for Altair
    corr_df = corr_matrix.reset_index().melt(id_vars='index')
    corr_df.columns = ['index_x', 'index_y', 'value']
    
    heatmap = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X('index_x:O', title='Features'),
        y=alt.Y('index_y:O', title='Features'),
        color=alt.Color('value:Q', scale=alt.Scale(scheme='blueorange'), title='Correlation'),
        tooltip=['index_x', 'index_y', 'value']
    ).properties(
        title=title,
        width=600,
        height=600
    ).configure_axis(
        labelFontSize=18,  # Adjust font size for axis labels
        titleFontSize=18   # Adjust font size for axis titles
    ).configure_legend(
        labelFontSize=18,  # Set font size for legend labels
        titleFontSize=18   # Set font size for legend title
    )
    return heatmap

@st.cache_resource
# Main function to display correlation matrices
class CorrelationMatrix:
    @staticmethod
    def show_correlation_matrix(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal):
        st.subheader("Correlation Matrices for Different Datasets")

        # Cache correlation matrices
        if "corr_mitbih_train" not in st.session_state:
            st.session_state.corr_mitbih_train = compute_correlation_matrix(mitbih_train.iloc[:, :-1])
        if "corr_mitbih_test" not in st.session_state:
            st.session_state.corr_mitbih_test = compute_correlation_matrix(mitbih_test.iloc[:, :-1])
        if "corr_ptbdb_normal" not in st.session_state:
            st.session_state.corr_ptbdb_normal = compute_correlation_matrix(ptbdb_normal.iloc[:, :-1])
        if "corr_ptbdb_abnormal" not in st.session_state:
            st.session_state.corr_ptbdb_abnormal = compute_correlation_matrix(ptbdb_abnormal.iloc[:, :-1])

        # Tabs for selecting the dataset
        tab1, tab2, tab3, tab4 = st.tabs([
            "MIT-BIH Train Dataset", 
            "MIT-BIH Test Dataset", 
            "PTBDB Normal Dataset", 
            "PTBDB Abnormal Dataset"
        ])

        # Create a two-column layout for each dataset
        for tab, corr_matrix, title, dataset_name in zip(
                [tab1, tab2, tab3, tab4],
                [st.session_state.corr_mitbih_train, st.session_state.corr_mitbih_test, st.session_state.corr_ptbdb_normal, st.session_state.corr_ptbdb_abnormal],
                ["MIT-BIH Train Dataset Correlation Matrix",
                 "MIT-BIH Test Dataset Correlation Matrix",
                 "PTBDB Normal Dataset Correlation Matrix",
                 "PTBDB Abnormal Dataset Correlation Matrix"],
                ["MIT-BIH Train Dataset", "MIT-BIH Test Dataset", "PTBDB Normal Dataset", "PTBDB Abnormal Dataset"]):

            with tab:
                col1, col2 = st.columns(2)
                
                with col1:
                    heatmap = plot_heatmap(corr_matrix, title)
                    st.altair_chart(heatmap)
                
                with col2:
                    #st.write(f"### Observations for {dataset_name}")
                    st.write("""
                    - **Overall Observations**:
                      - The correlation matrices across the datasets (MIT-BIH Train/Test, PTBDB Normal/Abnormal) show distinct patterns of feature relationships.
                      - A strong diagonal (correlation = 1) is observed, as expected, indicating self-correlation.
                      - Blocks of high positive or negative correlations among features suggest potential redundancies or groupings.

                    - **Dataset-Specific Patterns**:
                      - **MIT-BIH Train/Test**: Similar patterns in correlation indicate consistency between training and testing datasets.
                      - **PTBDB Normal/Abnormal**: 
                        - The normal dataset exhibits strong localized correlations, reflecting stable signal patterns.
                        - The abnormal dataset shows more dispersed correlations, possibly due to higher variability in abnormal signals.

                    - **Impact on Model**:
                      - High correlations may introduce redundancy, suggesting the potential for dimensionality reduction (e.g., PCA).
                      - Understanding feature relationships helps refine feature selection and improve model interpretability.

                    - **Discussion Point**:
                      - Highlighting and leveraging these patterns can aid in better preprocessing and feature engineering.
                    """)

