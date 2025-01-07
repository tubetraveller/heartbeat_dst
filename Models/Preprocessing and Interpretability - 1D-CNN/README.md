# Heartbeat Classification with Enhanced Interpretability

## Project Overview
This project focuses on heartbeat classification using deep learning techniques. The approach emphasizes **interpretability** by preprocessing the ECG signals such that individual phases of the cardiac cycle (e.g., P, QRS, T waves) align across signals. This preprocessing step ensures that the averaged signals can be meaningfully compared and interpreted in terms of physiological phases.

Additionally, two distinct preprocessing methods are implemented to shift and align the ECG signals based on specific features of the heart cycle. The resulting models are analyzed using **SHAP (SHapley Additive exPlanations)** to provide insights into the contribution of individual time points to the classification decision.

## Goal of the Project
The main objective is to develop an interpretable heartbeat classification model that not only achieves high classification performance but also provides insights into the decision-making process. By aligning ECG signals to highlight specific cardiac phases, the project aims to bridge the gap between accuracy and interpretability, ultimately supporting medical diagnosis.

## Dataset Information
The project uses the **MIT-BIH Arrhythmia Dataset**, a widely used dataset in medical research for ECG signal classification.

### MIT-BIH Arrhythmia Dataset:
- **Number of Samples**: 109,446
- **Number of Classes**: 5
- **Sampling Frequency**: 125Hz
- **Classes**:
  - Class 0: Normal (N)
  - Class 1: Supraventricular ectopic beat (S)
  - Class 2: Ventricular ectopic beat (V)
  - Class 3: Fusion beat (F)
  - Class 4: Unclassifiable (Q)
- **Source**: Physionet's MIT-BIH Arrhythmia Dataset

The dataset is preprocessed to ensure consistent signal length by padding or cropping all ECG signals to a fixed length of 188.

## Preprocessing Methods
Two different preprocessing methods are introduced. Both align the **R-wave (maximum amplitude)** of the ECG signals to a fixed position across all samples. The signals are shifted either left or right without altering their values. Specific rules ensure that truncated values are appropriately reinserted, maintaining the integrity of the original signal.
Both methods aim to standardize the signal alignment, allowing meaningful interpretation of the averaged ECG signals across classes.

## Model Architecture
The classification model is built using a **Residual 1D-CNN** architecture. The model leverages residual blocks to capture spatial and temporal patterns in the preprocessed ECG signals.

### Key Features of the Model:
- **Residual Blocks**: Enhance feature extraction and allow deeper architectures without degradation.
- **Conv1D Layers**: Capture local patterns in the ECG signals.
- **Dropout Regularization**: Mitigates overfitting.
- **Softmax Output**: Provides probabilities for multi-class classification.

## Interpretability with SHAP
To assess the interpretability of the model, **SHAP (SHapley Additive exPlanations)** values are computed for test samples. SHAP values quantify the contribution of each time step to the model's predictions, highlighting which phases of the cardiac cycle are most influential for the classification.

### SHAP Analysis Workflow:
1. **Background Sample Selection**: A subset of training data is used as the background for SHAP computation.
2. **Test Sample Analysis**: SHAP values are computed for a subset of test samples.
3. **Visualization**: Mean SHAP values are overlaid with averaged ECG signals to provide insights into the most critical features for classification.

## Results
The preliminary results indicate that aligning ECG signals to specific cardiac phases improves interpretability without compromising accuracy. Key evaluation metrics include:
- **Accuracy**: High classification performance across all classes.
- **SHAP Analysis**: Highlights the phases of the cardiac cycle most relevant for distinguishing between classes.

## How to Run the Project
To run the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/tubetraveller/heartbeat_dst.git
   cd Heart-Interpretability
   ```
2. **Install Dependencies**:
   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Conclusion
This project demonstrates the potential of combining data preprocessing techniques with explainable AI methods to enhance the interpretability of deep learning models for medical applications. By aligning ECG signals to cardiac phases and analyzing model predictions with SHAP, the approach provides both high performance and actionable insights for physicians.
