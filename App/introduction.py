import os
import streamlit as st

base_dir = os.path.dirname(os.path.abspath(__file__))
visualizations_dir = os.path.join(base_dir, "EDA", "visualizations")

normal_signal_path = os.path.join(visualizations_dir, "normal_ekg_signal.png")
individual_signal_mitbih_path = os.path.join(visualizations_dir, "individual_signal_mitbih.png")

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
    The number of samples in both collections is sufficient large for the meaningful training of deep neural networks.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Abstract")
    st.markdown("""
    <div class="justified-text">
    These two datasets have been used to explore heartbeat classification using classical machine learning and deep neural network architectures.
    The two databases focus on two problems respectively: 

    - **PTB-Diagnostic ECG Database**: distinction to be made between normal and pathological ECGs with a focus on myocardial infarction
    - **MIT-BIH Arrhythmia Database**: differentiation between normal heartbeats and 4 alternative rhythmological classifications
    <div class="justified-text">
    The basic structure of all signals is as follows:
                
    - Each signal consists of 188 time series points, whereby 1.5 seconds are mapped at a sampling frequency of 125Hz
    - The data points of all signals are normalised and take values between 0 and 1
    <div class="justified-text">          
    The aim of this project was to gain a comprehensive understanding of the data sets and then to implement various ML and deep learning algorithms in an exploratory manner, achieving a balance between performance and interpretability
    </div>
    """, unsafe_allow_html=True)

    st.subheader("The Electrocardiogram: ")
    st.markdown("""
    <div class="justified-text">
    An electrocardiogram (ECG or EKG) is a medical test that measures the heart's electrical activity to detect rhythm, rate, and potential abnormalities. It is performed by placing electrodes on the skin at specific points, which capture the heart's electrical signals and display them as a waveform for analysis.
    </div>
    """, unsafe_allow_html=True)
    st.image(normal_signal_path, caption="Example of a normal ECG signal", width=700, use_container_width=False)


    st.subheader("The PTB Diagnostic ECG Database")
    st.write("""
    - **Number of Samples**: 14,552  
    - **Number of Categories**: 2  
    - **Sampling Frequency**: 125Hz  
    - **Data Source**: Physionet's PTB Diagnostic ECG Database  
    - **Classes:**: ['Normal': 0, 'Pathological': 1]
    - **Remark**: All the samples are cropped, downsampled, and padded with zeroes if necessary to the fixed dimension of 188.
    """)

    st.subheader("Arrhythmia Dataset")
    st.write("""
    - **Number of Samples**: 109,446  
    - **Number of Categories**: 5  
    - **Sampling Frequency**: 125Hz  
    - **Data Source**: Physionet's MIT-BIH Arrhythmia Dataset  
    - **Classes**: ['Normal': 0, 'Supraventricular': 1, 'Ventricular': 2, 'Fusion': 3, 'Unclassified (including Pacemaker-Beats)': 4]
    - **Remark**: All the samples are cropped, downsampled, and padded with zeroes if necessary to the fixed dimension of 188.
    """)

    st.subheader("The Actual Signals")
    st.image(individual_signal_mitbih_path, caption="Example of an individual ECG signal from the MIT-BIH dataset", width=900, use_container_width=False)
