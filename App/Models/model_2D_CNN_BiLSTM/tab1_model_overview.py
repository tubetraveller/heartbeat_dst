import streamlit as st


def model_overview():
    st.subheader("Model Overview")

    # Add a radio button to switch sections
    section = st.radio(
        "Explore the sections below:",
        options=["Why 2D-CNN-BiLSTM", "Model Design", "Training and Validation Process"],
        horizontal=True
    )

    if section == "Why 2D-CNN-BiLSTM":
        st.markdown("""
        ### Why I Chose the 2D-CNN-BiLSTM with XGBoost Architecture for ECG Classification

        This model combines **2D-CNN** for spatial feature extraction and **BiLSTM** for temporal dependencies, followed by **XGBoost** for robust classification. Here's why this architecture is ideal for ECG classification:

        1. **Representation of ECG as 2D Data**:
            - ECG signals, originally 1D time-series data, are reshaped into **2D matrices** during preprocessing.
            - This transformation allows the model to analyze **spatial relationships** across the signal, such as the patterns between **P, QRS, and T waves**.

        2. **Spatial Feature Extraction with 2D-CNN**:
            - **Why 2D-CNN?**
                - Convolutional Neural Networks (**CNNs**) are highly effective at detecting **local spatial patterns** in data.
            - **Application to ECG**:
                - For ECG signals, CNN layers extract meaningful spatial relationships within a single heartbeat.
                - These spatial features enhance the model's ability to detect **abnormalities or distinct rhythms** that signify conditions like arrhythmias.

        3. **Temporal Dependency Modeling with BiLSTM**:
            - **Why BiLSTM?**
                - Long Short-Term Memory (**LSTM**) networks excel at capturing **temporal dependencies** in sequential data.
                - BiLSTM extends this by processing data in both **forward and backward directions**.
            - **Combination with 2D-CNN**:
                - Features extracted by the 2D-CNN are reshaped and fed into BiLSTM layers.
                - This integration allows the model to simultaneously learn **spatial (CNN)** and **temporal (BiLSTM)** dependencies in the ECG data.

        4. **Improved Feature Classification with XGBoost**:
            - **Why XGBoost?**
                - XGBoost is a robust **gradient-boosting classifier** that excels at leveraging extracted features for precise classification.
            - **How It Complements the Pipeline**:
                - After feature extraction by 2D-CNN and temporal modeling by BiLSTM, XGBoost refines these features for robust classification.
                - It ensures high accuracy and robustness in predicting five distinct ECG categories (**N, S, V, F, Q**).

        5. **Proven Success in Research**:
            - **Cheng et al. (2021)**:
                - Demonstrated the effectiveness of combining **CNN and BiLSTM** for ECG classification.
                - Highlighted how BiLSTM captures **temporal dependencies** crucial for ECG analysis.
            - **Teijeiro et al. (2018)**:
                - Showed that **XGBoost enhances the classification accuracy** when paired with deep learning feature extractors for ECG beat classification.

        ### Summary of Key Advantages:
        - **2D-CNN**: Extracts high-level **spatial features** from the reshaped ECG signals.
        - **BiLSTM**: Captures **sequential relationships** between heartbeats, essential for distinguishing normal and abnormal rhythms.
        - **XGBoost**: Utilizes extracted features for robust classification, achieving **high performance** across all five ECG categories.

        ### References:
        1. [Cheng, J., Zou, Q., & Zhao, Y. ***"ECG signal classification based on deep CNN and BiLSTM."*** BMC Medical Informatics and Decision Making 21, 365 (2021).](https://doi.org/10.1186/s12911-021-01736-y)
        2. [Teijeiro, T. et al., ***"ECG beat classification using deep learning and XGBoost."*** Physiological Measurement 39, 125005 (2018).](https://doi.org/10.1088/1361-6579/aad7e4)
        """)


    elif section == "Model Design":
        st.markdown("""
        ### Model Design
        The CNN-BiLSTM model is implemented as follows:
        - **Input Shape**: `(17, 11, 1)` (reshaped 2D ECG signal).
        - **Convolutional Layers**:
            - First layer: 256 filters, kernel size `(3, 3)`, ReLU activation, followed by max pooling and dropout.
            - Second layer: 128 filters, kernel size `(2, 2)`, ReLU activation, followed by max pooling and dropout.
        - **Reshaping Layer**:
            - Converts the output of the convolutional layers into a shape suitable for LSTM layers.
        - **BiLSTM Layers**:
            - First BiLSTM: 128 units with return sequences.
            - Second BiLSTM: 32 units for summarizing sequential information.
        - **Dense Layers**:
            - Fully connected layer with 128 units and ReLU activation.
            - Final output layer with 5 units (softmax activation for multi-class classification).
        """)

        # Add an expander for the additional code
        with st.expander("Show Code: 2D-CNN-BiLSTM"):
            st.code("""
            # Separate features and labels 
            X_train = mitbih_train.iloc[:, :-1].values
            y_train = mitbih_train.iloc[:, -1].values

            X_test = mitbih_test.iloc[:, :-1].values
            y_test = mitbih_test.iloc[:, -1].values

            # Define the 2D-CNN-BiLSTM Model
            def create_cnn_bilstm_model():
                inputs = Input(shape=(17, 11, 1))
                
                # Convolutional layers
                x = Conv2D(256, kernel_size=(3, 3), activation='relu')(inputs)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(0.25)(x)
                
                x = Conv2D(128, kernel_size=(2, 2), activation='relu')(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(0.25)(x)
                
                # Reshape the data to be suitable for BiLSTM
                x = Reshape((x.shape[1], -1))(x)
                
                # BiLSTM layers
                x = Bidirectional(LSTM(128, return_sequences=True))(x)
                x = Bidirectional(LSTM(32))(x)
                
                # Dense layers
                x = Dense(128, activation='relu')(x)
                x = Dropout(0.5)(x)
                
                # Output layer
                outputs = Dense(5, activation='softmax')(x)  # 5 classes for MIT-BIH
                
                # Build the model
                model = Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                return model
            """, language="python")
         # Display the model architecture diagram
        st.image(
            "./Models/2D_CNN_BiLSTM/cnn_bilstm_model.svg", 
            caption="2D CNN-BiLSTM Model Architecture", 
            use_container_width=True
        )

    elif section == "Training and Validation Process":
        st.markdown("""
        ### Training and Validation Process
        The training and validation process is designed to:
        - Address the **class imbalance** issue using SMOTE.
        - Ensure a robust evaluation using **Stratified K-Fold Cross-Validation**.
        - Integrate **2D-CNN-BiLSTM** with **XGBoost** for feature refinement and classification.

        This workflow ensures that the model is evaluated on diverse subsets of the data while maintaining the class distribution in each fold.

        ## Stratified K-Fold Process: Key Steps

        Below are the steps involved in the Stratified K-Fold process for training and evaluating the CNN-BiLSTM model:

        1. **Stratified K-Fold Setup**:
        - Utilizes `StratifiedKFold` to ensure each fold contains a representative proportion of classes, crucial for imbalanced datasets like MIT-BIH.

        2. **SMOTE Application**:
        - Applies Synthetic Minority Oversampling Technique (SMOTE) to the training fold for balancing the class distribution.

        3. **Data Standardization**:
        - Standardizes both the training and validation data for consistency and improved model performance.

        4. **CNN-BiLSTM Model Training**:
        - Trains a CNN-BiLSTM model on the SMOTE-applied, standardized, and reshaped training data.
        - Implements early stopping to avoid overfitting by monitoring validation loss.

        5. **Feature Extraction**:
        - Extracts features from the trained CNN-BiLSTM model using its penultimate dense layer (`model.layers[-3]`).

        6. **XGBoost Training**:
        - Trains an XGBoost classifier on the extracted features for better interpretability and performance.

        7. **Validation**:
        - Validates the XGBoost model on the extracted validation features and stores:
            - Predictions (`y_val_pred`)
            - True labels (`y_val_fold`).
        - Collects all predictions and true labels for evaluation across all folds.

        8. **Classification Report**:
        - Generates and prints a classification report for each fold to evaluate performance metrics (e.g., precision, recall, F1-score).

        9. **Memory Management**:
        - Frees up memory by clearing the TensorFlow session after each fold to avoid memory overflow.

        10. **Iteration Over Folds**:
            - Repeats the entire process for each fold in the Stratified K-Fold split to ensure robust model training and evaluation.
    """)
        with st.expander("Show Code: Training and Validation"):
            st.code("""
            # Setup Stratified K-Fold
            skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

            fold_no = 1
            batch_size = 128

            final_predictions = []
            final_true_labels = []
            final_probs = []  # To store probabilities for each fold

            for train_index, val_index in skf.split(X_train, y_train):
                print(f"Training fold {fold_no}...")

                # Split the data into training and validation sets for this fold
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                # Apply SMOTE to the training fold only
                smote = SMOTE(random_state=42)
                X_train_fold_smote, y_train_fold_smote = smote.fit_resample(X_train_fold, y_train_fold)

                # Standardize data
                scaler = StandardScaler()
                X_train_fold_smote = scaler.fit_transform(X_train_fold_smote)
                X_val_fold = scaler.transform(X_val_fold)

                # Reshape the data for Conv2D
                X_train_fold_reshaped = X_train_fold_smote.reshape(-1, 17, 11, 1)
                X_val_fold_reshaped = X_val_fold.reshape(-1, 17, 11, 1)

                # Flatten the validation data for SHAP (store after the loop)
                X_val_flat = X_val_fold_reshaped.reshape(X_val_fold_reshaped.shape[0], -1)

                # Create the CNN-BiLSTM Model
                model = create_cnn_bilstm_model()

                # Add early stopping
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                # Train the model
                history = model.fit(X_train_fold_reshaped, y_train_fold_smote,
                                    epochs=10,
                                    batch_size=batch_size,
                                    validation_data=(X_val_fold_reshaped, y_val_fold),
                                    callbacks=[early_stopping])

                # Step 4: Extract Features from CNN-BiLSTM Model for the Current Fold
                feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)
                
                X_train_features = feature_extractor.predict(X_train_fold_reshaped)
                X_val_features = feature_extractor.predict(X_val_fold_reshaped)
                
                # Step 5: Train XGBoost on Extracted Features for the Current Fold
                xgb_model = xgb.XGBClassifier(
                    objective='multi:softmax',
                    num_class=5,
                    eval_metric='mlogloss',
                    use_label_encoder=False
                )
                
                xgb_model.fit(X_train_features, y_train_fold_smote)

                # Evaluate the XGBoost model on the validation set
                y_val_pred = xgb_model.predict(X_val_features)
                y_val_prob = xgb_model.predict_proba(X_val_features)  # Get probabilities for ROC

                final_predictions.extend(y_val_pred)
                final_true_labels.extend(y_val_fold)
                final_probs.extend(y_val_prob)  # Append probabilities for the current fold

                # Print classification report for the current fold
                print(f"Classification Report for Fold {fold_no}:",classification_report(y_val_fold, y_val_pred, target_names=['N', 'S', 'V', 'F', 'Q']))

                # Clear the session to free up memory
                tf.keras.backend.clear_session()

                fold_no += 1
            """, language="python")
        
