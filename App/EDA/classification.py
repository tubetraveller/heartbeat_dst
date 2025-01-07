import streamlit as st
import pandas as pd
import altair as alt

# Function to display the EDA page
class EDA:
    @staticmethod
    def show_class_distribution(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal):
        st.header("Class Distribution Across Datasets")
        st.markdown("     ")

        # Combine PTBDB datasets for unified class distribution
        ptbdb_combined = pd.concat([ptbdb_normal, ptbdb_abnormal])

        # Class mappings
        mitbih_mapping = {
            0: "N",
            1: "S",
            2: "V",
            3: "F",
            4: "Q"
        }

        ptbdb_mapping = {
            0: "Normal",
            1: "Abnormal"
        }

        # Tabs for selecting the dataset
        tab1, tab2, tab3 = st.tabs(["MIT-BIH Train Dataset", "MIT-BIH Test Dataset", "PTBDB Combined Dataset"])

        # Helper function to plot class distribution with Altair
        def plot_class_distribution(data, title, mapping):
            data = data.copy()  # Create a copy of the dataset to avoid modifying the original
            data[187] = data[187].map(mapping)  # Map the class labels using the provided mapping
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
                labelFontSize=16,
                titleFontSize=16
            ).configure_legend(
                labelFontSize=16,    # Set font size for legend labels
                titleFontSize=16     # Set font size for legend title
            )
            return chart

        # Tab 1: MIT-BIH Train Dataset
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                chart = plot_class_distribution(
                    mitbih_train, "Class Distribution in MIT-BIH Train Dataset", mitbih_mapping
                )
                st.altair_chart(chart)
            with col2:
                st.write("""
                - **Class Imbalance**:
                  - Class N dominates the dataset (~70,000 samples).
                - **Minority Classes**:
                  - Classes S and F have significantly fewer samples.
                - **Impact**:
                  - Model may favor Class N during predictions.
                  - Minority classes could face challenges in recall.
                - **Potential Approach**:
                  - Techniques like SMOTE or oversampling could improve balance.
                  - Evaluation metrics such as F1-Score and AUC-ROC can provide better insights into model performance.
                """)

        # Tab 2: MIT-BIH Test Dataset
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                chart = plot_class_distribution(
                    mitbih_test, "Class Distribution in MIT-BIH Test Dataset", mitbih_mapping
                )
                st.altair_chart(chart)
            with col2:
                st.write("""
                - **Consistency in Class Imbalance**:
                  - The test dataset mirrors the imbalance seen in the training dataset, with Class N being dominant and minority classes (S, F) underrepresented.
                - **Potential Impact on Generalization**:
                  - The model's ability to generalize to minority classes might be hindered due to this imbalance in both training and test datasets.
                - **Model Performance Expectation**:
                  - Evaluation may favor high performance for Class N but struggle with accurate predictions for minority classes.
                - **Follow-up Discussion**:
                  - If balancing techniques were applied during training, results on the test set can demonstrate their effectiveness for minority classes.
                """)

        # Tab 3: PTBDB Combined Dataset
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                chart = plot_class_distribution(
                    ptbdb_combined, "Class Distribution in PTBDB Combined Dataset", ptbdb_mapping
                )
                st.altair_chart(chart)
            with col2:
                st.write("""
                - **Class Distribution**:
                  - The dataset contains two classes: **Normal** and **Abnormal**.
                  - Class **Abnormal** has approximately **10,000 samples**, making it the majority class.
                  - Class **Normal** has around **4,000 samples**, making it the minority class.
                - **Class Imbalance**:
                  - There is an imbalance, but it's less severe compared to the MIT-BIH dataset.
                  - The ratio between Abnormal and Normal is approximately **2.5:1**.
                - **Potential Impact**:
                  - Models trained on this dataset might slightly favor Class Abnormal due to its higher representation.
                  - Despite the imbalance, the dataset still offers a reasonable distribution to train on both classes with appropriate techniques.
                - **Discussion Point**:
                  - Balancing techniques, if applied, could further improve recall for Class Normal without compromising overall model performance.
                """)
