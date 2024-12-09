import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Function to compute correlation matrix and cache the result
@st.cache_data
def compute_correlation_matrix(df):
    return df.corr()

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

# Main function to display correlation matrices
class CorrelationMatrix:
    @staticmethod
    def show_correlation_matrix(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal):
        #st.header("Exploratory Data Analysis (EDA)")
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

        # Shared commentary for each dataset
        shared_commentary = """
        The figure presents correlation matrices for four heartbeat datasets: MIT-BIH Train, MIT-BIH Test, PTBDB Normal, and PTBDB Abnormal. The color gradient from blue to orange indicates the strength of these correlations.

        The MIT-BIH Train and Test datasets show similar correlation patterns, suggesting consistency between training and testing data, which is crucial for developing reliable predictive models. The strong diagonal correlations indicate that features are more strongly correlated with themselves, as expected. The PTBDB Normal dataset shows lower overall correlations, while the PTBDB Abnormal dataset displays higher correlations, indicating more pronounced and interrelated features in abnormal heartbeats.

        These insights can refine diagnostic algorithms, making them more adept at distinguishing between normal and abnormal heartbeats, thereby reducing misdiagnosis and improving patient outcomes. By focusing on highly correlated features in the abnormal dataset, healthcare providers can streamline data processing, leading to significant cost savings and increased efficiency. More accurate diagnostics can lead to quicker and more reliable patient outcomes, enhancing overall patient care and satisfaction.

        Based on this analysis, it is recommended to reduce dimensionality by focusing on highly correlated features within the abnormal dataset. This can streamline the model without compromising its diagnostic power, improving both efficiency and accuracy. Additionally, leveraging these correlation insights during model training will ensure that the model learns the most significant patterns for both normal and abnormal heartbeats, leading to better generalization and predictive performance. Further research should be conducted to understand the underlying reasons for the observed correlations, potentially uncovering new biomarkers for heart conditions and leading to the development of even more effective diagnostic tools.

        By translating these technical findings into actionable business insights, healthcare organizations can enhance their data-driven strategies, leading to improved diagnostic solutions, cost efficiencies, and better patient outcomes.
        """

        # Create a button to show or hide the code block before each chart
        for tab, corr_matrix, title, dataset_name in zip(
                [tab1, tab2, tab3, tab4],
                [st.session_state.corr_mitbih_train, st.session_state.corr_mitbih_test, st.session_state.corr_ptbdb_normal, st.session_state.corr_ptbdb_abnormal],
                ["MIT-BIH Train Dataset Correlation Matrix",
                 "MIT-BIH Test Dataset Correlation Matrix",
                 "PTBDB Normal Dataset Correlation Matrix",
                 "PTBDB Abnormal Dataset Correlation Matrix"],
                ["MIT-BIH Train Dataset", "MIT-BIH Test Dataset", "PTBDB Normal Dataset", "PTBDB Abnormal Dataset"]):

            with tab:
                # Checkbox to show code
                show_code = st.checkbox(f"Show {dataset_name} Code", key=f"show_code_{dataset_name}")
                if show_code:
                    st.code(f"""
import altair as alt

# Transform correlation matrix to long-form for Altair
corr_df = corr_{dataset_name.lower().replace(' ', '_')}.reset_index().melt(id_vars='index')
corr_df.columns = ['index_x', 'index_y', 'value']

heatmap = alt.Chart(corr_df).mark_rect().encode(
    x=alt.X('index_x:O', title='Features'),
    y=alt.Y('index_y:O', title='Features'),
    color=alt.Color('value:Q', scale=alt.Scale(scheme='blueorange'), title='Correlation'),
    tooltip=['index_x', 'index_y', 'value']
).properties(
    title='{title}',
    width=600,
    height=600
)
st.altair_chart(heatmap)
                    """)
                # Display the heatmap
                heatmap = plot_heatmap(corr_matrix, title)
                st.altair_chart(heatmap)
                # Shared commentary
                st.write(shared_commentary)

