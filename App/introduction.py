import streamlit as st

# Inject CSS for text justification
st.markdown("""
    <style>
    .justified-text {
        text-align: justify;
    }
    </style>
    """, unsafe_allow_html=True)

# Function for the initial page (Dataset Introduction)
def introduction_page():
    st.title("ECG Heartbeat Categorization Dataset")
    
    st.subheader("Context")
    st.markdown("""
    <div class="justified-text">
    The ECG Heartbeat Categorization Dataset is composed of two collections of heartbeat signals 
    derived from two famous datasets in heartbeat classification: the MIT-BIH Arrhythmia Dataset and the PTB Diagnostic ECG Database. 
    The number of samples in both collections is large enough for training deep neural networks.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Abstract")
    st.markdown("""
    <div class="justified-text">
    This dataset has been used to explore heartbeat classification using deep neural network architectures and to observe some of the capabilities of transfer learning. 
    The signals correspond to electrocardiogram (ECG) shapes of heartbeats for both normal cases and cases affected by different arrhythmias and myocardial infarction. 
    These signals are preprocessed and segmented, with each segment corresponding to a heartbeat.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Arrhythmia Dataset")
    st.write("""
    - **Number of Samples**: 109,446  
    - **Number of Categories**: 5  
    - **Sampling Frequency**: 125Hz  
    - **Data Source**: Physionet's MIT-BIH Arrhythmia Dataset  
    - **Classes**: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
    """)

    st.subheader("The PTB Diagnostic ECG Database")
    st.write("""
    - **Number of Samples**: 14,552  
    - **Number of Categories**: 2  
    - **Sampling Frequency**: 125Hz  
    - **Data Source**: Physionet's PTB Diagnostic ECG Database  
    - **Remark**: All the samples are cropped, downsampled, and padded with zeroes if necessary to the fixed dimension of 188.
    """)
