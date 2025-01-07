import os
import pandas as pd
import streamlit as st

# Dynamically determine the base directory (one level up from the current script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dynamically construct the path to the Data folder
DATA_DIR = os.path.join(BASE_DIR, 'Data')

@st.cache_data
def load_datasets():
    """Load datasets from the Data directory."""
    mitbih_train = pd.read_csv(os.path.join(DATA_DIR, 'mitbih_train.csv'), header=None)
    mitbih_test = pd.read_csv(os.path.join(DATA_DIR, 'mitbih_test.csv'), header=None)
    ptbdb_abnormal = pd.read_csv(os.path.join(DATA_DIR, 'ptbdb_abnormal.csv'), header=None)
    ptbdb_normal = pd.read_csv(os.path.join(DATA_DIR, 'ptbdb_normal.csv'), header=None)
    
    return mitbih_train, mitbih_test, ptbdb_abnormal, ptbdb_normal

def display_data_loading_code():
    """Display the code block for data loading."""
    st.code("""
import pandas as pd

# Load the datasets
mitbih_train = pd.read_csv('mitbih_train.csv', header=None)
mitbih_test = pd.read_csv('mitbih_test.csv', header=None)
ptbdb_abnormal = pd.read_csv('ptbdb_abnormal.csv', header=None)
ptbdb_normal = pd.read_csv('ptbdb_normal.csv', header=None)
    """, language="python")

def generate_data_audit(dataset, dataset_name):
    """Generate a data audit report for the given dataset."""
    audit_data = []
    for col in dataset.columns:
        col_data = {
            '# Column': col,
            'Name of the Column': f'Feature {col}' if col < dataset.shape[1] - 1 else 'Label',
            'Variable\'s type': 'Feature' if col < dataset.shape[1] - 1 else 'Target',
            'Description': f'ECG signal feature {col + 1}' if col < dataset.shape[1] - 1 else 'ECG class label',
            'Is the variable available before prediction': 'Yes' if col < dataset.shape[1] - 1 else 'No',
            'Variable\'s type (detailed)': str(dataset.dtypes[col]),  # Convert dtype to string
            'Percentage of missing values': f"{dataset[col].isnull().mean() * 100:.2f}%",
            'Categorical / Quantitative': 'Quantitative' if col < dataset.shape[1] - 1 else 'Categorical'
        }
        audit_data.append(col_data)
    
    audit_df = pd.DataFrame(audit_data)

    # Display the DataFrame in Streamlit
    st.subheader(f"{dataset_name} Dataset")
    st.dataframe(audit_df)

def data_audit(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal): 
    """Perform data audit based on user-selected dataset."""
    dataset_labels = ["Train", "Test", "Normal", "Abnormal"]
    dataset_option = st.select_slider("Select Dataset to Audit", options=dataset_labels)

    if dataset_option == "Train":
        generate_data_audit(mitbih_train, "MIT-BIH Train")
    elif dataset_option == "Test":
        generate_data_audit(mitbih_test, "MIT-BIH Test")
    elif dataset_option == "Normal":
        generate_data_audit(ptbdb_normal, "PTBDB Normal")
    elif dataset_option == "Abnormal":
        generate_data_audit(ptbdb_abnormal, "PTBDB Abnormal")
