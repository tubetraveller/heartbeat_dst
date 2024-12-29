# Heartbeat Classification Project

## Overview
This project aims to classify heartbeats using the ECG Heartbeat Categorization Dataset. The dataset comprises heartbeat signals extracted from two well-known sources: the MIT-BIH Arrhythmia Dataset and the PTB Diagnostic ECG Database. The PTB Diagnostic ECG database allows a distinction to be made between normal and pathological ECGs with a focus on myocardial infarction. The MIT-BIH Arrythmia database allows differentiation between normal heartbeats and 4 alternative rhythmological classifications. The signals allow detailed statistical and visualisation analysis as well as the possibility of implementing various machine learning and deep learning algorithms for classification. 

## Dataset
The dataset used in this project is from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?select=ptbdb_normal.csv), containing preprocessed and segmented ECG heartbeat signals:

1. **MIT-BIH Arrhythmia Dataset**: Contains 5 classes of heartbeat signals labeled as follows:
   - N (Normal Beat)
   - S (Supraventricular Ectopic Beat)
   - V (Ventricular Ectopic Beat)
   - F (Fusion of Ventricular and Normal Beat)
   - Q (Unclassifiable Beat)

2. **PTB Diagnostic ECG Database**: Contains heartbeats labeled as either normal or abnormal.

The signals are sampled at 125 Hz, and each segment represents one heartbeat, with padding to ensure consistent length across samples.
All signals are already normalised and assume values between 0 and 1. Each signal begins with the sensing of a QRS complex which serves as an annotation signal. The signal ends after the next complete QRS complex - in other words, it comprises one more incomplete QRS complex than a regular cardiac cycle. 

## Project Structure
The repository is organized as follows:

- **App**: Contains the Streamlit app for interactive visualization and analysis. The app includes modules for data loading, auditing, and exploratory data analysis (EDA).
- **EDA**: Contains Jupyter notebooks used for EDA, focusing on understanding the data distribution, visualizing class distribution, and identifying useful trends.
- **Models**: Includes two different machine learning models used to classify the heartbeat data.

## Approaches
This project leverages multiple approaches for data exploration and model building:

1. **Data Loading and Auditing**: The data loading and initial auditing phase ensures the datasets are in the correct format and allows for a detailed overview of features, missing values, and data types.

2. **Exploratory Data Analysis (EDA)**: Detailed EDA is conducted using:
   - **Individual Signal Analysis**: To get an overview of the different classes and unterstand the korrelation to a "regular" heart-cycle. 
   - **Class Distribution**: To understand the balance of different classes within the datasets, visualized using bar charts.
   - **Rolling Mean Visualization**: To analyze trends across feature values, providing insights into the characteristics of different heartbeats.
   - **Relationship between averaged and individual signal**: To understand the relationship of the averaged signal to the individual signals and to create the basis for interpretability. 

3. **Machine Learning Models**: Multiple machine learning models are developed to classify the heartbeats, focusing on techniques such as:
   - **Bidirectional Long Short-Term Memory (BiLSTM)**: Combined with CNN layers to capture both spatial features and sequential dependencies in the ECG signals. The CNN layers extract spatial features, while the BiLSTM layers capture temporal relationships across different segments of the heartbeat signal.
   - **Interpretability and Preprocessing - 1D-CNN**: An approach to enable interpretability of the model through extended data preprocessing. The classification model is built using a Residual 1D-CNN architecture. The model leverages residual blocks to capture spatial and temporal patterns in the preprocessed ECG signals.
     
4. **Model Interpretability**: Approaches like SHAP and LIME are applied to understand the feature importance and interpret the models' decisions.

## Getting Started
To get started with this project:

1. Clone the repository:
   ```
   git clone https://github.com/tubetraveller/heartbeat_dst.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run App/app.py
   ```

## Data Sources
- **MIT-BIH Arrhythmia Dataset**: Available at [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?select=ptbdb_normal.csv).
- **PTB Diagnostic ECG Database**: Available at [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?select=ptbdb_normal.csv).

## Usage
- **EDA**: Explore the data using the Jupyter notebooks provided in the `EDA` folder.
- **App**: Use the Streamlit app to visualize the data interactively and get insights.
- **Model Training**: The `Models` folder contains scripts for training machine learning models for heartbeat classification.

## License
This project is licensed under the MIT License. 
## Acknowledgments
We would like to thank [Kaggle](https://www.kaggle.com) for providing the datasets used in this project.

