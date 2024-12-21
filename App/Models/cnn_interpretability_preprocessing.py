import os
import streamlit as st

# Dynamisch den Pfad relativ zu app.py erstellen
base_dir = os.path.dirname(os.path.abspath(__file__))
visualizations_dir = os.path.join(base_dir, "visualizations_1D_CNN")

# Bilderpfade
model_architecture_path = os.path.join(visualizations_dir, "model_architecture.png")
classification_report_orig_path = os.path.join(visualizations_dir, "classification_report_orig.png")
confusion_matrix_orig_path = os.path.join(visualizations_dir, "confusion_matrix_orig.png")
normal_signal_path = os.path.join(visualizations_dir, "normal_ekg_signal.png")
individual_signal_path = os.path.join(visualizations_dir, "individual_signal_mitbih.png")
average_signals_per_class_train_path = os.path.join(visualizations_dir, "average_signals_per_class_train.png")
original_vs_preprocessed_signal_path = os.path.join(visualizations_dir, "original_vs_preprocessed_signal.png")
average_signal_preprocessed_path = os.path.join(visualizations_dir, "average_signal_preprocessed_with_background.png")
classification_report_preprocessed_path = os.path.join(visualizations_dir, "classification_report_preprocessed.png")
confusion_matrix_preprocessed_path = os.path.join(visualizations_dir, "confusion_matrix_preprocessed.png")


def show_1d_cnn_interpretability_preprocessing():
    st.title("Interpretability through Preprocessing - 1D-CNN")

    # Untermenü
    subpage = st.sidebar.radio(
        "Sections",
        ["The Model", "The Problem", "Preprocessing", "Interpretability"]
    )

    if subpage == "The Model":
        st.subheader("Model Architecture")
        st.image(model_architecture_path, caption="Model Architecture", use_container_width=True)
        if st.checkbox("Show Model Code"):
            code = """
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import numpy as np

# Residual Block Definition
def residual_block(x, filters, kernel_size=3):
    # Shortcut connection
    shortcut = x

    # Adjust shortcut dimensions if necessary
    if x.shape[-1] != filters:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(shortcut)
    
    # First Conv1D layer
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second Conv1D layer
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Add the shortcut to the output
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

# Model input
inputs = Input(shape=(187, 1), name="Input")  # Shape adjusted according to preprocessing

# 1D-CNN layer with the best hyperparameters
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = BatchNormalization()(x)

# Residual block 1
x = residual_block(x, filters=64)

# Residual block 2
x = residual_block(x, filters=128)

# Residual block 3
x = residual_block(x, filters=256)

# Flatten layer
x = Flatten()(x)

# Dense layers with the best dropout values
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)  # Best dropout for layer 1

x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)  # Best dropout for layer 2

# Output layer for multi-class classification
output_layer = Dense(5, activation='softmax')(x)

# Create the model
model = Model(inputs=inputs, outputs=output_layer)

# Compile the model with the best learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model Summary:")
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model
training_history = model.fit(
    X_train_orig,
    y_train_orig,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)
            """
            st.code(code, language="python")
        st.markdown(
            """
### **Model Architecture:** 
**This model is a 1D Convolutional Neural Network (1D-CNN) architecture designed for multi-class classification with residual 
connections to improve feature learning especially for sequential Data.** 

**Key Components of the Model:**

1. Input Layer:
- The input shape is (187, 1), representing a time-series signal with 187 timesteps and a single feature. This korresponds to the EKG-Data.

2. Convolutional and Pooling Layers:
- The first layer is a Conv1D layer with 64 filters, kernel size of 3, and ReLU activation.
- A MaxPooling1D layer with a pool size of 2 is applied to downsample the input

3. Residual Blocks:
- The model includes three residual blocks, each consisting of:
    - Two Conv1D layers with the same filter size and kernel size of 3.
    - BatchNormalization layers after each convolution to stabilize and speed up training.
    - ReLU activation for non-linearity.
    - A shortcut connection adds the input to the block's output to form a residual connection.
    - Filter sizes for the residual blocks:
        - Block 1: 64 filters
        - Block 2: 128 filters
        - Block 3: 256 filters

4. Flatten Layer:
- The output of the last residual block is flattened into a single vector with 23,808 elements to connect it to fully connected layers.

5. Fully Connected Dense Layers:
- The dense layers progressively reduce the feature dimensions:
    - First dense layer: 128 neurons with ReLU activation and a dropout of 40% to prevent overfitting.
    - Second dense layer: 64 neurons with ReLU activation and a dropout of 30% to prevent overfitting.

6. Output Layer:
- A dense layer with 5 neurons and a softmax activation function for multi-class classification. The model outputs probabilities for each of the 5 classes.

**Total Parameters:**
- The model has 3,449,469 trainable parameters, primarily concentrated in the dense layers and residual blocks.

**Optimizer and Loss Function:**
- The model uses the Adam optimizer with a learning rate of 0.0001 for adaptive gradient descent.
- The loss function is categorical_crossentropy, suitable for multi-class classification tasks.
"""
        )
        st.subheader("Results on Original Data")
        st.image(classification_report_orig_path, caption="Classification Report on Original Data", use_container_width=True)
        st.image(confusion_matrix_orig_path, caption="Confusion Matrix on Original Data", use_container_width=True)
        st.markdown(
            """
The results of this model are quite satisfactory with an overall accuracy of 99% and a precision for all classes of over 90%. 
            The biggest weakness is the recall for minority classes 1 and 3.
            """
        )

    elif subpage == "The Problem":
        st.subheader("Normal EKG-Signal")
        st.image(normal_signal_path, caption="Normal EKG Signal", use_container_width=True)
        st.markdown(
            """
The graph shows a regular ECG signal. 
The signal begins with a P wave, followed by a QRS complex and finally the T wave. It is important to note that the shape of the T wave is determined by the QRS complex as it is sequential data and the T wave reflects the regression of excitation of the heart. 
            """
        )
        st.subheader("Individual Signal from the MIT-BIH Database")
        st.image(individual_signal_path, caption="Individual Signal from MIT-BIH Database", use_container_width=True)
        st.markdown(
            """  
We can see that the Signal (here as an example a normal heartbeat (Class:0)) deviates from the regular ECG signal in the following way:
- In the column with index 0, the signal starts as an annotation signal, triggered by the maximum of a QRS complex. The signal therefore begins with an incomplete QRS complex (actually only RS-Complex).
- This is followed by the T wave, which represents the regression of the excitation of the incompletely imaged QRS complex.
- If present, the P-wave then occurs.
- This is followed by the actual QRS complex which needs to be analyzed. 
This means: To determine the heart rhythm (class 0-4) a T-wave is used which is not related to the QRS complex which is primarily analyzed. 
            """
        )
        st.subheader("Averaged Signals per Class")
        st.image(average_signals_per_class_train_path, caption="Averaged Signals per Class", use_container_width=True)
        st.markdown(
            """
The graph shows the averaged ECG signals for the MIT-BIH-Train database divided into the different diagnostic classes. It can be seen that even if there are clear differences between the averaged signals for the various classes, they bear no resemblance to a “normal” ECG signal. 
The reason for this phenomenon is as follows: The R-R intervals vary greatly depending on the respective heart rate. Since rhythm (class 0-4) and heart rate are largely independent of each other, the second QRS complex of a signal occurs at completely different column indices. For the P-QRS interval, the dependence on the heart rate is much less pronounced. I use this phenomenon in relation to the following manipulation of the data.
            """
        )

    elif subpage == "Preprocessing":
        st.subheader("Advanced Preprocessing")
        if st.checkbox("Show Code for preprocess_ecg_signals_1"):
            preprocess_code = """
def preprocess_ecg_signals_1(df, peak_col_index=32):
    '''
    - Preprocess ECG signals to align the maximum value (R-wave) to the same column.
    - All values from the original signal remain unchanged and are only shifted.
    - In the case of a left shift, the truncated values are inserted at the point where zero-padding begins,
    but only if there are at least 10 consecutive zeros from that point onward.
    - If the first value of the truncated values differs by more than 0.05 from the last value before padding,
    the truncated values are adjusted by the difference.
    - No values are altered or duplicated.
    - Additionally, the number of left shifts, right shifts, and no shifts are counted.

    Parameters:
    - df: DataFrame containing ECG signals, with labels in the last column.
    - peak_col_index: Index of the column where the maximum value (R-wave) should be aligned (after column removal).

    Returns:
    - df_preprocessed: Preprocessed DataFrame.
    - shift_counts: Dictionary containing the number of left, right, and no shifts.
    '''

    # 1. Remove the first 22 columns
    df = df.drop(df.columns[0:22], axis=1).reset_index(drop=True)

    # Signal columns (excluding the label column)
    signal_columns = df.columns[:-1]

    # Total number of signal columns
    num_cols = len(signal_columns)

    # Initialize shift counters
    left_shift_count = 0
    right_shift_count = 0
    no_shift_count = 0

    # Process each row
    for idx in df.index:
        signal = df.loc[idx, signal_columns].values

        # 2. Identify the maximum value (R-wave)
        max_index = np.argmax(signal)

        # 3. Calculate the shift
        shift = peak_col_index - max_index  # Positive: right shift, Negative: left shift

        if shift > 0:
            # Right shift
            right_shift_count += 1

            end_part = signal[-shift:]       # Last 'shift' elements
            main_part = signal[:-shift]      # Remaining signal

            # Create the new signal
            new_signal = np.concatenate((end_part, main_part))

        elif shift < 0:
            # Left shift
            left_shift_count += 1

            shift_abs = -shift
            main_part = signal[shift_abs:]   # Signal starting from 'shift_abs'
            start_part = signal[:shift_abs]  # First 'shift_abs' elements

            # Find the starting position of padding (minimum 10 consecutive zeros)
            def find_padding_start(sig, min_zero_length=10):
                zero_runs = (sig == 0).astype(int)
                convolved = np.convolve(zero_runs, np.ones(min_zero_length, dtype=int), mode='valid')
                padding_starts = np.where(convolved == min_zero_length)[0]
                if len(padding_starts) > 0:
                    return padding_starts[0]
                else:
                    return None

            zero_index = find_padding_start(main_part, min_zero_length=10)
            if zero_index is not None:
                # Insert the truncated values at the padding location
                before_padding = main_part[:zero_index]
                after_padding = main_part[zero_index:]

                # Check the difference between signal values
                if len(before_padding) > 0:
                    last_value_before_padding = before_padding[-1]
                    first_value_of_start_part = start_part[0]

                    difference = first_value_of_start_part - last_value_before_padding

                    if abs(difference) > 0.05:
                        # Adjust the truncated values
                        start_part = start_part - difference

                # Create the new signal
                new_signal = np.concatenate((before_padding, start_part, after_padding))
            else:
                # No padding found, append truncated values at the end
                new_signal = np.concatenate((main_part, start_part))
        else:
            # No shift required
            no_shift_count += 1
            new_signal = signal

        # Ensure the signal has the original length
        if len(new_signal) < num_cols:
            # Pad with zeros at the end
            new_signal = np.concatenate((new_signal, np.zeros(num_cols - len(new_signal))))
        elif len(new_signal) > num_cols:
            # Truncate the signal to the original length
            new_signal = new_signal[:num_cols]

        # Update the DataFrame
        df.loc[idx, signal_columns] = new_signal

    # Create a dictionary with shift counts
    shift_counts = {
        'left_shifts': left_shift_count,
        'right_shifts': right_shift_count,
        'no_shifts': no_shift_count
    }

    return df.reset_index(drop=True), shift_counts
            """
            st.code(preprocess_code, language="python")
        st.markdown(
            """
### **What does `preprocess_ecg_signals_1` do?**

1. **Purpose**: Preprocess ECG signals to align the maximum value (R-wave) to a specified column while preserving all original values.

2. **Key Features**:
   - **Column removal**: Removes the first 22 columns to eliminate duplication and spikes for better alignment.
   - **R-peak alignment**: Aligns the R-wave to a specified column (`peak_col_index`).

3. **Steps**:
   - **R-peak detection**: Identifies the maximum value (R-wave) in the signal after removing the first 22 columns.
   - **Shift calculation**:
     - Positive shifts: Right shift of the signal.
     - Negative shifts: Left shift of the signal.
   - **Signal shifting**:
     - **Right Shift**: Moves the end part of the signal to the front, aligning the R-wave with the specified column.
     - **Left Shift**:
       - Locates where padding with at least 10 consecutive zeros begins.
       - Reinserts the truncated signal values at the padding position if possible.
       - Adjusts truncated signal values if the difference between the first truncated value and the last value before padding exceeds 0.05.
       - Moves the start of the signal to the padding position, preserving all original values.
   - **Signal length consistency**:
     - Ensures the processed signal retains the original length by:
       - Padding with zeros if the signal is shorter.
       - Truncating the signal if it becomes longer.

4. **Tracking shifts**:
   - Counts the number of left shifts, right shifts, and no shifts performed for reporting.

5. **Output**:
   - **Preprocessed DataFrame**: Contains the shifted ECG signals.
   - **Shift counts**: A dictionary with counts of left, right, and no shifts performed.  
            """
        )
        st.subheader("Comparison of Original and Preprocessed Signal")
        st.image(original_vs_preprocessed_signal_path, caption="Original vs Preprocessed Signal", use_container_width=True)
        st.subheader("Results of Averaged EKG-Signal by Class after `preprocess_ecg_signals_1`")
        st.image(average_signal_preprocessed_path, caption="Averaged Signals After Preprocessing", use_container_width=True)
        st.markdown(
            """
### **Advantages and Disadvantages of `preprocess_ekg_signals_1`:**
- Advantages: 
   - The resulting averaged signal is highly consistent with an individual ECG signal. 
   - The assignment is marked in color in the illustration. 
   
- Disadvantages: 
   - To display only one ECG cycle, 22 columns are deleted. This reduces the information content. 
   - Over 90% of all signals are shifted to the left. The T wave is therefore added to the QRS complex to which it does not actually belong.
            """
        )
        st.subheader("Modeling with `preprocess_ecg_signals_1`")
        st.image(classification_report_preprocessed_path, caption="Classification Report with Preprocessed Data", use_container_width=True)
        st.image(confusion_matrix_preprocessed_path, caption="Confusion Matrix with Preprocessed Data", use_container_width=True)
        st.markdown(
            """
### **Interpretation:**
As expected, the modeling results are worse with the pre-processed data, which is expressed in particular in a reduction of the Precision and recall for the minority-classes (1 and 3).However, a reduced classification performance for precision and recall can be observed for all classes. 
            """
        )

    elif subpage == "Interpretability":
        st.subheader("Interpretability Through Visualization and Analysis")
        
        # Explanation section
        st.markdown(
            """
### **Why Interpretability Matters**
Understanding the decision-making process of a model is crucial for trust, transparency, and debugging. 
In the context of ECG signals, interpretability can help ensure that the model focuses on relevant patterns, 
such as the P wave, QRS complex, and T wave, rather than spurious correlations in the data.
            """
        )

        # Mean EKG-Data by Class and SHAP-Values
        st.subheader("Mean EKG-Data by Class and SHAP-Values")
        st.markdown(
            """
SHAP values provide a consistent measure of the contribution of each feature (or data point in the time series) 
to the model's predictions. This helps identify which parts of the ECG signal are most influential for specific diagnostic classes. 
The fact that we use pre-processed data allows us to establish a link between the results of the SHAP analysis 
and the different phases of the cardiac cycle.
            """
        )

        # Combined signals and SHAP visualization
        combined_signals_and_shap_path = os.path.join(visualizations_dir, "combined_signals_and_shap.png")
        st.image(combined_signals_and_shap_path, caption="Mean EKG Signals and SHAP Values", use_container_width=True)

        st.markdown(
            """
### **Description of the Results:**

The visualized plot shows the **mean EKG signals** and their corresponding **SHAP values** for the five different classes. For each class, two stacked plots are presented:

1. **Mean EKG Signal (Blue Line):**
   - Displays the average amplitude of the signal across all time steps for a given class.
   - Each line represents the averaged signal from all samples in the respective class.

2. **SHAP Values (Color-Coded Bars):**
   - Highlights the importance of each time step in determining the class predictions.
   - **Red Bars:** Positive SHAP values, indicating a stronger contribution to the class prediction.
   - **Blue Bars:** Negative SHAP values, indicating less relevance or contribution against the prediction.
   - The intensity of the color represents the magnitude of the SHAP value.

#### **Interpretation:**

- **SHAP Values:**
  - The SHAP values identify key time regions that significantly influence the classification.
  - For instance:
    - For class 1, high SHAP values are shown in columns 21-23 - correlating with the P-wave which, as expected, should not occur here.
    - While the positive and negative predictive SHAP values for classes 2 and 3 are concentrated around the P and QRS wave, there is a wider spread for classes 1 and 4 in particular, so that the T wave was also relevant in the classification.
  - The SHAP values effectively highlight the time steps that are most critical for distinguishing between classes.

#### **What is the added value of these findings?**
- The **interpretability of the model**, by showing which parts of the signal are most influential for classification decisions.
- <u>**For the first time, it is now possible to directly link the behavior of the model to the individual phases of the cardiac cycle.**</u>

#### **What could be further improved?**
- Due to long calculation times, the “background_size” and “test_sample_size” for the DeepExplainer are relatively small at 1000 in the current example.
- Due to the considerable class imbalances in the data set, the significance of the results is limited, particularly with regard to the underrepresented classes (“SVEB” and “Fusion”).
            """
        )

        # Top 15 Feature Importances per Class
        st.subheader("Top 15 Feature Importances per Class")
        top_shap_features_per_class_path = os.path.join(visualizations_dir, "top_shap_features_per_class.png")
        st.image(top_shap_features_per_class_path, caption="Top 15 SHAP Feature Importances per Class", use_container_width=True)

        # Summary
        st.markdown(
            """
### **Summary**
In this section, we presented a deep learning model to classify the rhythm of heartbeats. Through further preprocessing, 
we try to make the model accessible for interpretability so that the decision-making process is comprehensible for medical professionals. 
This does not affect the model performance too much. We believe that the interpretability of classification decisions will be the basis 
for acceptance and the clinical implementation of artificial intelligence, especially in the healthcare sector.
            """
        )
