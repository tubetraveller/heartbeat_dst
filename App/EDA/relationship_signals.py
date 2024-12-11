import os
import streamlit as st

# Dynamisch den Pfad relativ zu app.py erstellen
base_dir = os.path.dirname(os.path.abspath(__file__))
visualizations_dir = os.path.join(base_dir, "visualizations")

# Bilderpfade
mit_bih_signal_korrelation_normal_path = os.path.join(visualizations_dir, "mit_bih_signal_korrelation_normal.png")
ptbdb_signal_korrelation_abnormal_path = os.path.join(visualizations_dir, "PTBDB_signal_korrelation_abnormal.png")
mean_ekg_data_mit_bih_path = os.path.join(visualizations_dir, "mean_ekg_data_mit_bih.png")
mean_data_ptbdb_path = os.path.join(visualizations_dir, "mean_data_PTBDB.png")
distribution_first_row_path = os.path.join(visualizations_dir, "MITBIH_dist_first_row.png")
qq_plots_path = os.path.join(visualizations_dir, "hist_qq_MITBIH.png")

def show_relationship_signals():
    st.title("Relationship Between Averaged and Individual Signal")

    # Erster Abschnitt mit Tabs
    st.subheader("Combined Graphs for Individual and Averaged Signals")
    tab1, tab2 = st.tabs(["MIT-BIH Dataset", "PTBDB Dataset"])

    with tab1:
        st.image(mit_bih_signal_korrelation_normal_path, caption="Combined Graph - MIT-BIH Dataset", use_container_width=True)

    with tab2:
        st.image(ptbdb_signal_korrelation_abnormal_path, caption="Combined Graph - PTBDB Dataset", use_container_width=True)

    st.markdown(
        """
        The graphs in this section are combined graphs and refer to one of the two datasets (PTBDB, MIT-BIH). 
        First, 4 exemplary individual ECG signals are shown, then the mean value over the lines of all ECG signals is shown. 
        The values here all refer to ECG signals that were classified as either normal (MIT-BIH-Dataset) or abnormal (PTBDB-Dataset). 
        This selection was purely random and is intended to illustrate that the observations apply to all diagnostic classes of both datasets (MIT-BIH and PTBDB).

        The P wave, QRS complex, and T wave can be clearly differentiated in the individual signals. 
        The basic pattern is a data peak in columns 0-3, followed by a rapid drop. This data peak is a consequence of the preprocessing 
        of the data and is not part of the actual ECG signal. The actual ECG signal begins with the start of the T wave, with a data peak 
        on average around data points 25-50. As a consequence of different heart rates, the P wave and QRS complex appear at different data points. 
        After the QRS complex, the filling with zeros occurs at different data points as well - also as a result of the preprocessing of the data. 
        This is also not part of the actual ECG signal.

        These observations help in understanding the averaged ECG signal for normal heartbeats. For the MIT-BIH-Dataset (Diagnostic Class: Normal), 
        there is a high signal at the beginning which then decreases continuously, with peaks around data points 25-40 as well as 60 and 80. 
        While the individual signals are therefore quite diagnostic, this is no longer the case for the averaged data points. 
        The only part of the signal that can still be traced reliably using the averaged values is the T-wave.
        """
    )

    # Zweiter Abschnitt mit zwei Bildern untereinander
    st.subheader("Mean EKG Signals")
    st.image(mean_ekg_data_mit_bih_path, caption="Averaged Signals - MIT-BIH Dataset", use_container_width=True)
    st.image(mean_data_ptbdb_path, caption="Averaged Signals - PTBDB Dataset", use_container_width=True)

    st.markdown(
        """
        The graphs show the averaged signals of all corresponding classes. While there are clear differences between the various 
        classes of the averaged signals, the actual ECG signal is no longer clearly discernible for any of the classes.

        Furthermore, the data peak in the first columns of each signal, as well as all averaged signals, is noticeable. 
        The first column serves as an annotation signal for the previous QRS complex. Most signals were assigned the value 1 here.
        The lower graphs show histograms for the data distribution in column 1. With such a large overlap, it can be assumed 
        that the diagnostic discriminatory power of this row itself is low.
        """
    )

    # Dritter Abschnitt mit Verteilungen
    st.subheader("Distribution of the First Row")
    st.image(distribution_first_row_path, caption="Distribution of the First Row", use_container_width=True)
    st.markdown(
        """
        This observation is also statistically well substantiated. Using D'Agostino Pearson's Normality Test, all columns—separated by diagnostic 
        categories—were tested for normal distribution. The significance level was set at alpha = 0.05. Consequently, for the PTBDB dataset, 
        none of the columns, for the MIT-BIH Train dataset, 0.21% of the columns, and for the MIT-BIH Test dataset, 1.06% of the columns can 
        be assumed to follow a normal distribution. These results were obtained even though the tests for normal distribution were performed 
        separately for the respective diagnostic classes. 

        Thus, for the vast majority of columns, a normal distribution cannot be assumed. Given that we are assessing physiological data from 
        humans (and have already pre-sorted by diagnostic classes), this is certainly a noteworthy fact. The preprocessing of the data is 
        likely a significant influence here as well.

        The lower graphic depicts the distribution of the data for an exemplary column along with the corresponding QQ plot. 
        For the displayed column and diagnostic classes, normal distribution cannot be assumed.
        """
    )
    st.image(qq_plots_path, caption="QQ Plot and Distribution for an Exemplary Column", use_container_width=True)

