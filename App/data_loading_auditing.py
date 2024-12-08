import pandas as pd
import streamlit as st

# Load the datasets with caching to prevent repeated loading
@st.cache_data
def load_datasets():
    mitbih_train = pd.read_csv(r'C:\Users\maxgl\heartbeat_dst_working_copy\heartbeat_dst\Models\Preprocessing and Interpretability - 1D-CNN\mitbih_train.csv', header=None)
    mitbih_test = pd.read_csv(r'C:\Users\maxgl\heartbeat_dst_working_copy\heartbeat_dst\Models\Preprocessing and Interpretability - 1D-CNN\mitbih_test.csv', header=None)
    ptbdb_abnormal = pd.read_csv(r'C:\Users\maxgl\heartbeat_dst_working_copy\heartbeat_dst\Models\Preprocessing and Interpretability - 1D-CNN\ptbdb_abnormal.csv', header=None)
    ptbdb_normal = pd.read_csv(r'C:\Users\maxgl\heartbeat_dst_working_copy\heartbeat_dst\Models\Preprocessing and Interpretability - 1D-CNN\ptbdb_normal.csv', header=None)

    return mitbih_train, mitbih_test, ptbdb_abnormal, ptbdb_normal

# Function to display the Data Loading code block without reloading
def display_data_loading_code():
    #st.subheader("Data Loading")
    st.code("""
import pandas as pd

# Load the datasets
mitbih_train = pd.read_csv('mitbih_train.csv', header=None)
mitbih_test = pd.read_csv('mitbih_test.csv', header=None)
ptbdb_abnormal = pd.read_csv('ptbdb_abnormal.csv', header=None)
ptbdb_normal = pd.read_csv('ptbdb_normal.csv', header=None)
    """, language="python")

# Function to generate the data audit table for a given dataset
def generate_data_audit(dataset, dataset_name):
    audit_data = []
    for col in dataset.columns:
        col_data = {
            '# Column': col,
            'Name of the Column': f'Feature {col}' if col < dataset.shape[1] - 1 else 'Label',
            'Variable\'s type': 'Feature' if col < dataset.shape[1] - 1 else 'Target',
            'Description': f'ECG signal feature {col + 1}' if col < dataset.shape[1] - 1 else 'ECG class label',
            'Is the variable available before prediction': 'Yes' if col < dataset.shape[1] - 1 else 'No',
            'Variable\'s type (detailed)': dataset.dtypes[col],
            'Percentage of missing values': f"{dataset[col].isnull().mean() * 100:.2f}%",
            'Categorical / Quantitative': 'Quantitative' if col < dataset.shape[1] - 1 else 'Categorical'
        }
        audit_data.append(col_data)
    
    audit_df = pd.DataFrame(audit_data)
    st.subheader(f"{dataset_name} Dataset")
    st.dataframe(audit_df)

def data_audit(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal): 
    # Dataset selection using a select slider instead of radio buttons
    dataset_labels = ["Train", "Test", "Normal", "Abnormal"]
    dataset_option = st.select_slider("Select Dataset to Audit", options=dataset_labels)

    # Generate the data audit table for the selected dataset
    if dataset_option == "Train":
        generate_data_audit(mitbih_train, "MIT-BIH Train")
    elif dataset_option == "Test":
        generate_data_audit(mitbih_test, "MIT-BIH Test")
    elif dataset_option == "Normal":
        generate_data_audit(ptbdb_normal, "PTBDB Normal")
    elif dataset_option == "Abnormal":
        generate_data_audit(ptbdb_abnormal, "PTBDB Abnormal")

