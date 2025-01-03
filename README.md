# Heartbeat Classification Project

## Overview
This project aims to classify heartbeats using the ECG Heartbeat Categorization Dataset. The dataset comprises heartbeat signals extracted from two well-known sources: the MIT-BIH Arrhythmia Dataset and the PTB Diagnostic ECG Database. These signals provide insights into both normal and abnormal heartbeats, which can be used to develop various machine learning models and perform exploratory data analysis.

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

## Project Structure
The repository is organized as follows:

- **App**: Contains the Streamlit app for interactive visualization and analysis. The app includes modules for data loading, auditing, and exploratory data analysis (EDA).
- **EDA**: Contains Jupyter notebooks used for EDA, focusing on understanding the data distribution, visualizing class distribution, and identifying useful trends.
- **Models**: Includes different machine learning models used to classify the heartbeat data.

## Approaches
This project leverages multiple approaches for data exploration and model building:

1. **Data Loading and Auditing**: The data loading and initial auditing phase ensures the datasets are in the correct format and allows for a detailed overview of features, missing values, and data types.

2. **Exploratory Data Analysis (EDA)**: Detailed EDA is conducted using:
   - **Class Distribution**: To understand the balance of different classes within the datasets, visualized using bar charts.
   - **Rolling Mean Visualization**: To analyze trends across feature values, providing insights into the characteristics of different heartbeats.

3. **Machine Learning Models**: Multiple machine learning models are developed to classify the heartbeats, focusing on techniques such as:
   - **Short Models explanation (CNN)**: by Max  
   - **Bidirectional Long Short-Term Memory (BiLSTM)**: Combined with CNN layers to capture both spatial features and sequential dependencies in the ECG signals. The CNN layers extract spatial features, while the BiLSTM layers capture temporal relationships across different segments of the heartbeat signal. 
     
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

