
import os
import streamlit as st

# Dynamisch den Pfad relativ zu app.py erstellen
base_dir = os.path.dirname(os.path.abspath(__file__))
visualizations_dir = os.path.join(base_dir, "visualizations")

# Bilderpfade
normal_signal_path = os.path.join(visualizations_dir, "normal_ekg_signal.png")
individual_signal_mitbih_path = os.path.join(visualizations_dir, "individual_signal_mitbih.png")
ptbdb_signals_path = os.path.join(visualizations_dir, "ptbdb_signals.png")
mitbih_signals_path = os.path.join(visualizations_dir, "mitbih_signals.png")

def show_individual_signals():
    st.title("The Individual Signals")
    
    # Normal EKG-Signal
    st.subheader("Normal EKG-Signal")
    st.image(normal_signal_path, caption="Normal EKG Signal", use_container_width=True)
    st.markdown(
        """
        The graph shows a regular ECG signal. 
        The signal begins with a P wave, followed by a QRS complex and finally the T wave. It is important to note that the shape of the T wave 
        is determined by the QRS complex as it is sequential data and the T wave reflects the regression of excitation of the heart.
        """
    )
    
    # Individual Signal from the MIT-BIH Database
    st.subheader("Individual Signal from the MIT-BIH Database")
    st.image(individual_signal_mitbih_path, caption="Individual Signal from the MIT-BIH Database", use_container_width=True)
    st.markdown(
        """
        We can see that the signal (here as an example a normal heartbeat (Class: 0)) deviates from the regular ECG signal in the following way:
        
        - In the column with index 0, the signal starts as an annotation signal, triggered by the maximum of a QRS complex. The signal therefore begins with an incomplete QRS complex (actually only RS-complex).
        - This is followed by the T wave, which represents the regression of the excitation of the incompletely imaged QRS complex.
        - If present, the P-wave then occurs.
        - This is followed by the actual QRS complex which needs to be analyzed.
        
        **This means:** To determine the heart rhythm (class 0-4), a T-wave is used which is not related to the QRS complex which is primarily analyzed.
        """
    )

    # PTBDB-Grafik
    st.subheader("PTBDB Signals")
    st.image(ptbdb_signals_path, caption="Visualization of Random EKG Signals - PTBDB", use_container_width=True)
    st.markdown(
        """
        This visualization showcases random EKG signals from the PTBDB dataset. 
        It includes examples from both normal and pathological classes, providing a visual comparison.
        """
    )

    # MITBIH-Grafik
    st.subheader("MITBIH Signals")
    st.image(mitbih_signals_path, caption="Visualization of Random EKG Signals - MITBIH", use_container_width=True)
    st.markdown(
        """
        This visualization presents random EKG signals from the MIT-BIH dataset. 
        It includes examples from all five diagnostic classes, illustrating the diversity of EKG signals within the dataset.
        This visualization only serves to give an impression of how exemplary signals of the different classes of datasets can look 
        and what the basic structure of the data is. Even at first glance, clear differences between the various signals can be identified. 
        It is possible for a physician to use these signals to arrive at the same classification as the one already given.
        """
    )
