import os
import streamlit as st

# Set the page layout to wide
st.set_page_config(layout="wide")

from data_loading_auditing import load_datasets, display_data_loading_code, data_audit  # Import functions
from introduction import introduction_page  # Import introduction page function
from EDA.classification import EDA  # Import the EDA class for classification
from EDA.correlation_matrix import CorrelationMatrix  # Import the Correlation Matrix class


# Get the absolute path to the directory containing this script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the CSS file relative to this script
css_file_path = os.path.join(base_dir, 'assets', 'styles.css')

# Load the CSS file
with open(css_file_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Outline")
page = st.sidebar.selectbox("Please choose a page", ["Introduction", "Data Loading & Auditing", "EDA", "Models", "Demo"])

# Cache datasets when navigating to the "Data Loading & Auditing" or "EDA" page
if 'datasets_loaded' not in st.session_state:
    with st.spinner("Loading datasets..."):
        mitbih_train, mitbih_test, ptbdb_abnormal, ptbdb_normal = load_datasets()
        st.session_state['datasets_loaded'] = (mitbih_train, mitbih_test, ptbdb_abnormal, ptbdb_normal)

# Retrieve cached data
mitbih_train, mitbih_test, ptbdb_abnormal, ptbdb_normal = st.session_state['datasets_loaded']

# Page logic based on selection
if page == "Introduction":
    introduction_page()

elif page == "Data Loading & Auditing":
    st.title("Data Loading & Auditing")

    # Create tabs for "Data Loading" and "Data Auditing"
    tab1, tab2 = st.tabs(["Data Loading", "Data Auditing"])

    with tab1:
        display_data_loading_code()

    with tab2:
        data_audit(mitbih_train, mitbih_test, ptbdb_abnormal, ptbdb_normal)

elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # Sidebar Navigation for EDA Sub-pages
    eda_subpage = st.sidebar.selectbox("Choose an EDA Sub-page", ["Class Distribution", "Correlation Matrix", "Individual Signals", "Signal Relationship",])

    # Sub-page logic for EDA
    if eda_subpage == "Class Distribution":
        eda = EDA()  # Create an instance of the EDA class for classification
        eda.show_class_distribution(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal)

    elif eda_subpage == "Correlation Matrix":
        correlation_matrix = CorrelationMatrix()  # Create an instance of the CorrelationMatrix class
        correlation_matrix.show_correlation_matrix(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal)

    elif eda_subpage == "Individual Signals":
        from EDA.individual_signals import show_individual_signals
        show_individual_signals()

    elif eda_subpage == "Signal Relationship":
        from EDA.relationship_signals import show_relationship_signals
        show_relationship_signals()

elif page == "Models":
    st.title("Deep Learning Models")
    model_subpage = st.sidebar.selectbox(
        "Choose a Model Sub-page",
        ["1D CNN Interpretability and Preprocessing", "2D CNN-BiLSTM"]
    )

    if model_subpage == "1D CNN Interpretability and Preprocessing":
        from Models.cnn_interpretability_preprocessing import show_1d_cnn_interpretability_preprocessing
        show_1d_cnn_interpretability_preprocessing()

    elif model_subpage == "2D CNN-BiLSTM":
        from Models.model_2D_CNN_BiLSTM.CNN_BiLSTM_2D import show_2d_cnn_bilstm
        show_2d_cnn_bilstm()

elif page == "Demo":
    st.title("Model Demo")

    # Sidebar for Demo Pages
    demo_subpage = st.sidebar.selectbox(
        "Choose a Demo", 
        ["2D CNN-BiLSTM"]
    )

    # Add logic for each demo
    if demo_subpage == "2D CNN-BiLSTM":
        from Demos.demo_2D_CNN_BiLSTM import show_demo_2d_cnn_bilstm
        show_demo_2d_cnn_bilstm()

