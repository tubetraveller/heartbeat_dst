import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Function to display the EDA page
class EDA:
    @staticmethod
    def show_class_distribution(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal):
        st.header("Class Distribution Across Datasets")
        st.markdown("     ")

        # Combine PTBDB datasets for unified class distribution
        ptbdb_combined = pd.concat([ptbdb_normal, ptbdb_abnormal])

        # Tabs for selecting the dataset
        tab1, tab2, tab3 = st.tabs(["MIT-BIH Train Dataset", "MIT-BIH Test Dataset", "PTBDB Combined Dataset"])

        # Helper function to plot with Altair
        def plot_class_distribution(data, title):
            class_counts = data[187].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            
            chart = alt.Chart(class_counts).mark_bar().encode(
                x=alt.X('Class:N', title='Class'),
                y=alt.Y('Count:Q', title='Count'),
                color=alt.Color('Class:N', scale=alt.Scale(scheme='tableau20'))  # Use color scheme for different shades
            ).properties(
                title=title,
                width=600,
                height=400
            ).configure_axis(
                labelFontSize=18,
                titleFontSize=18
            ).configure_legend(
                labelFontSize=18,    # Set font size for legend labels
                titleFontSize=18     # Set font size for legend title
            )
            return chart

        with tab1:
            #st.write("### MIT-BIH Train Dataset")
            st.code("""
import altair as alt

class_counts = mitbih_train[187].value_counts().reset_index()
class_counts.columns = ['Class', 'Count']

chart = alt.Chart(class_counts).mark_bar().encode(
    x=alt.X('Class:N', title='Class'),
    y=alt.Y('Count:Q', title='Count'),
    color=alt.Color('Class:N', scale=alt.Scale(scheme='tableau20'))  # Use color scheme for different shades
).properties(
    title='Class Distribution in MIT-BIH Train Dataset',
    width=600,
    height=400
)
st.altair_chart(chart)
""")
            chart = plot_class_distribution(mitbih_train, 'Class Distribution in MIT-BIH Train Dataset')
            st.altair_chart(chart)
            st.write("""
The figure presents the class distribution for two heartbeat datasets: MIT-BIH Train and MIT-BIH Test. The classes are labeled as N (Normal), S (Supraventricular ectopic beat), V (Ventricular ectopic beat), F (Fusion of ventricular and normal beat), and Q (Unclassifiable beat). The MIT-BIH Train dataset is heavily skewed towards the Normal (N) class, which constitutes the majority of the dataset. There is a notable presence of V and Q classes, while the S and F classes are underrepresented. Similarly, the MIT-BIH Test dataset shows the Normal (N) class as the most dominant. The distribution of other classes (S, V, F, Q) is consistent with the training dataset, indicating a balanced representation between training and testing data.

The class imbalance, particularly the dominance of the Normal (N) class, can affect model training. Models may become biased towards the majority class, potentially leading to poor performance in detecting minority classes. Ensuring balanced representation or using techniques such as class weighting or oversampling can help mitigate this issue, leading to more robust and accurate diagnostic models. Given the skewed distribution, more resources may need to be allocated to improve the detection of less represented classes (S, V, F, Q). This focus is critical for developing comprehensive diagnostic tools that can accurately identify all types of heartbeats. Accurate and balanced detection of all heartbeat types is crucial for reliable patient diagnosis and care. The skewed distributions highlight the need for targeted efforts to improve the detection of rare but clinically significant heartbeats. Enhanced diagnostic tools can lead to better patient outcomes by enabling timely and accurate identification of various cardiac conditions, reducing the risk of misdiagnosis.

To address class imbalance, it is recommended to implement techniques such as class weighting, oversampling of minority classes, or undersampling of the majority class in the MIT-BIH datasets. Data augmentation methods can also be considered to increase the representation of underrepresented classes. Developing and training models specifically tailored to improve the detection of less represented classes (S, V, F, Q) ensures a comprehensive diagnostic approach. Utilizing ensemble methods to combine different models can enhance overall performance and accuracy. Further research should be conducted to understand the clinical significance of each class and its impact on patient outcomes. Investing in advanced diagnostic tools that leverage the balanced detection of all classes can improve overall healthcare delivery.

By addressing class imbalances and focusing on improving diagnostic capabilities for all heartbeat types, healthcare organizations can enhance their data-driven strategies, leading to better diagnostic solutions, cost efficiencies, and improved patient outcomes.
""")

        with tab2:
            #st.write("### MIT-BIH Test Dataset")
            st.code("""
import altair as alt

class_counts = mitbih_test[187].value_counts().reset_index()
class_counts.columns = ['Class', 'Count']

chart = alt.Chart(class_counts).mark_bar().encode(
    x=alt.X('Class:N', title='Class'),
    y=alt.Y('Count:Q', title='Count'),
    color=alt.Color('Class:N', scale=alt.Scale(scheme='tableau20'))  # Use color scheme for different shades
).properties(
    title='Class Distribution in MIT-BIH Test Dataset',
    width=600,
    height=400
)
st.altair_chart(chart)
""")
            chart = plot_class_distribution(mitbih_test, 'Class Distribution in MIT-BIH Test Dataset')
            st.altair_chart(chart)

        with tab3:
            #st.write("### PTBDB Combined Dataset")
            st.code("""
import altair as alt

class_counts = ptbdb_combined[187].value_counts().reset_index()
class_counts.columns = ['Class', 'Count']

chart = alt.Chart(class_counts).mark_bar().encode(
    x=alt.X('Class:N', title='Class'),
    y=alt.Y('Count:Q', title='Count'),
    color=alt.Color('Class:N', scale=alt.Scale(scheme='tableau20'))  # Use color scheme for different shades
).properties(
    title='Class Distribution in PTBDB Combined Dataset',
    width=600,
    height=400
)
st.altair_chart(chart)
""")
            chart = plot_class_distribution(ptbdb_combined, 'Class Distribution in PTBDB Combined Dataset')
            st.altair_chart(chart)
