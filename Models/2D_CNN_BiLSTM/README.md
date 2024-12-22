# Heartbeat Classification Using CNN-BiLSTM Model

## Project Overview
This project focuses on classifying heartbeats using a deep neural network model based on the **CNN-BiLSTM** architecture. The goal of the project is to classify ECG signals into several categories, including normal and abnormal heartbeats, by leveraging deep learning techniques. The dataset used for this project contains heartbeat signals derived from the **MIT-BIH Arrhythmia Dataset** and the **PTB Diagnostic ECG Database**.

## Goal of the Project
The main objective of this project is to develop a robust heartbeat classification model that can effectively differentiate between normal heartbeats and those affected by different arrhythmias and myocardial infarction. This model aims to contribute to the broader field of medical diagnosis by aiding in the identification of potentially dangerous heart conditions.

## Dataset Information
The dataset consists of two collections of ECG (Electrocardiogram) signals, both widely used in medical research for heartbeat classification.

### MIT-BIH Arrhythmia Dataset:
- **Number of Samples**: 109,446
- **Number of Classes**: 5
- **Sampling Frequency**: 125Hz
- **Classes**: ['N': 0 (Normal), 'S': 1 (Supraventricular ectopic beat), 'V': 2 (Ventricular ectopic beat), 'F': 3 (Fusion beat), 'Q': 4 (Unclassifiable)]
- **Source**: Physionet's MIT-BIH Arrhythmia Dataset

### PTB Diagnostic ECG Database:
- **Number of Samples**: 14,552
- **Number of Classes**: 2
- **Sampling Frequency**: 125Hz
- **Classes**: ['N': 0 (Healthy control), 'MI': 1 (Myocardial infarction)]
- **Source**: Physionet's PTB Diagnostic Database

Both datasets have been preprocessed and segmented, with each segment corresponding to a single heartbeat. All samples are zero-padded or cropped to a fixed length of 188 to maintain consistency.

### Data Files:
The dataset is available as a collection of CSV files. Each CSV file contains a matrix where:
- Each row represents an individual heartbeat signal.
- The final column in each row denotes the class label.

## Model Architecture
The model used for heartbeat classification is a combination of **Convolutional Neural Networks (CNN)** and **Bidirectional Long Short-Term Memory (BiLSTM)** networks. The CNN layers capture spatial features from the ECG signal, while the BiLSTM layers capture the sequential dependencies between different segments of the signal.

**Model Diagram**:  
(Ongoing Development)

The architecture is designed to leverage both the temporal and spatial aspects of the ECG data to improve classification accuracy.

## Results
The model achieved promising results in classifying the ECG signals across different heartbeat categories. The key evaluation metrics include:
- **Accuracy**: 0.98 
- **F1 Score**: 0.99
- **Confusion Matrix**: (Ongoing Development)
- **App**:(Ongoing Development)

## How to Run the Project
To run this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/scriptsaso/Heart.git
   cd Heart
Placeholder for model details
