import streamlit as st
from dataloading_audit import load_datasets, display_data_loading_code, data_audit  # Import functions
from EDA import EDA  # Import the EDA class

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Introduction", "Data Loading & Auditing", "EDA"])

# Cache datasets when navigating to the "Data Loading & Auditing" or "EDA" page
if 'datasets_loaded' not in st.session_state:
    with st.spinner("Loading datasets..."):
        mitbih_train, mitbih_test, ptbdb_abnormal, ptbdb_normal = load_datasets()
        st.session_state['datasets_loaded'] = (mitbih_train, mitbih_test, ptbdb_abnormal, ptbdb_normal)

# Retrieve cached data
mitbih_train, mitbih_test, ptbdb_abnormal, ptbdb_normal = st.session_state['datasets_loaded']

# Page logic based on selection
if page == "Introduction":
    from introduction import introduction_page
    introduction_page()

elif page == "Data Loading & Auditing":
    st.title("Data Loading & Auditing")

    # Checkboxes to toggle between sections
    show_data_loading = st.checkbox("Show Data Loading")
    show_data_audit = st.checkbox("Show Data Auditing")

    # Display the selected sections based on checkboxes
    if show_data_loading and not show_data_audit:
        display_data_loading_code()

    elif show_data_audit and not show_data_loading:
        data_audit(mitbih_train, mitbih_test, ptbdb_abnormal, ptbdb_normal)

    elif show_data_loading and show_data_audit:
        st.warning("Please select only one section at a time.")

    if not show_data_loading and not show_data_audit:
        st.write("Please select a section to display.")

elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # Sidebar Navigation for EDA Sub-pages
    eda_subpage = st.sidebar.selectbox("Choose an EDA Sub-page", ["Class Distribution", "Box Plot of Features", "Correlation Matrix", "ECG Signal Examples"])

    eda = EDA()  # Create an instance of the EDA class

    # Sub-page logic for EDA
    if eda_subpage == "Class Distribution":
        eda.show_class_distribution(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal)

    elif eda_subpage == "Box Plot of Features":
        eda.show_box_plot(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal)

    elif eda_subpage == "Correlation Matrix":
        eda.show_correlation_matrix(mitbih_train)

    elif eda_subpage == "ECG Signal Examples":
        eda.show_ecg_signal_examples(mitbih_train)
