import streamlit as st

def output_classes():
    st.subheader("Output Classes")
    st.markdown("""
    The model classifies ECG signals into the following categories:
    - **N**: Normal
    - **S**: Supraventricular ectopic beat
    - **V**: Ventricular ectopic beat
    - **F**: Fusion of ventricular and normal beat
    - **Q**: Unclassifiable beat
    """)
