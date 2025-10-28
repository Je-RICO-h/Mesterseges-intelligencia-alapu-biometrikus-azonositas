# The baseline is from the "TypeNet: Deep Learning Keystroke Biometrics" publication
# The architecture is reconstructed from their thesis reference: https://github.com/BiDAlab/TypeNet, https://arxiv.org/abs/2101.05570

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, BatchNormalization, Dropout, Masking, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import time

USER_ID_COLUMN = 'label' 
TARGET_SEQUENCE_LENGTH = 5 # The thesis states that 5 length was used
MAIN_USER_LABEL = "User"
LSTM_UNITS = 128
LEARNING_RATE = 0.0001
PATIENCE_VALUE = 8 


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
FEATURE_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Trained_Model', 'Baseline', 'typenet_feature_extractor.h5'))
SCALER_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Trained_Model', 'Baseline', 'typenet_minmax_scaler.pkl'))
FEATURE_LIST_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Trained_Model', 'Baseline', 'typenet_feature_list.pkl'))
PROCESSED_FILE_NAME = os.path.abspath(os.path.join(BASE_DIR, '..', 'Data_Processed', 'data_processed.csv'))


def is_corrupted(column_name):
    return '\\x' in column_name or '\\u' in column_name


def preprocess_and_scale_data(file_path):
    df = pd.read_csv(file_path)
    
    #Deal with corrupted columns (5 to be exact)
    clean_columns = [
        col for col in df.columns if not is_corrupted(col)
    ]

    df = df[clean_columns].copy()
    
    #Drop timestamps, because we don't need them
    df = df.drop(columns=['start_time_timestamp', 'end_time_timestamp'], errors='ignore')
    
    df[USER_ID_COLUMN] = df[USER_ID_COLUMN].apply(
        lambda x: MAIN_USER_LABEL if x == MAIN_USER_LABEL else "Anomaly"
    )

    #Convert the type
    df['error'] = df['error'].astype(int)
    df['is_burst'] = df['is_burst'].astype(int)

    #Create categorical columns
    df_encoded = pd.get_dummies(df, columns=['key', 'combination'], dtype=int)
    
    #Separate the label column
    df_labels = df_encoded[USER_ID_COLUMN]
    df_features = df_encoded.drop(columns=[USER_ID_COLUMN])
    
    #Deal with missing data
    #'end_time'
    if 'end_time' in df_features.columns:
        prev_values_et = df_features['end_time'].ffill()
        next_values_et = df_features['end_time'].bfill()
        missing_indices_et = df_features['end_time'].isnull()
        df_features.loc[missing_indices_et, 'end_time'] = \
            prev_values_et[missing_indices_et].fillna(0) + next_values_et[missing_indices_et].fillna(0)

    #Interpolation of missing hold datas and speed
    interp_cols = ['variation_in_typing_speed', 'hold_time', 'interval']
    for col in interp_cols:
        if col in df_features.columns:
            df_features[col] = df_features[col].interpolate(method='linear', limit_direction='both')
            
    #NAN Cleanup (Not needed at this point, but just to be safe)
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    df_processed = pd.concat([df_features, df_labels], axis=1)

    final_feature_list = df_features.columns.tolist() 

    #Separate data into anomaly and main data
    main_user_data = df_processed[df_processed[USER_ID_COLUMN] == MAIN_USER_LABEL].drop(USER_ID_COLUMN, axis=1)
    other_users_data = df_processed[df_processed[USER_ID_COLUMN] != MAIN_USER_LABEL].drop(USER_ID_COLUMN, axis=1)

    #Scale the data
    scaler = MinMaxScaler()
    scaler.fit(main_user_data.values) # Fit only on main user data

    # Transform both data sets
    main_user_scaled = scaler.transform(main_user_data.values)
    other_users_scaled = scaler.transform(other_users_data.values)
    
    # Reassemble the data with labels
    X_scaled = np.vstack([main_user_scaled, other_users_scaled])
    Y_labels_raw = pd.concat([
        df_processed[df_processed[USER_ID_COLUMN] == MAIN_USER_LABEL][USER_ID_COLUMN], 
        df_processed[df_processed[USER_ID_COLUMN] != MAIN_USER_LABEL][USER_ID_COLUMN]
    ]).values
    
    N_FEATURES_PER_TIMESTEP = X_scaled.shape[1]
    
    return X_scaled, Y_labels_raw, N_FEATURES_PER_TIMESTEP, scaler, final_feature_list


def create_short_sequences(user_data, user_label, sequence_length = TARGET_SEQUENCE_LENGTH):
    """
    Create sequences from the big dataset
    """
    sequences = []
    labels = []
    total_length = user_data.shape[0]
    step = 1
    
    for start in range(0, total_length - sequence_length + 1, step):
        end = start + sequence_length
        sequences.append(user_data[start:end, :])
        labels.append(user_label)
        
    return sequences, labels


def read_and_prepare_keystroke_data_full(file_path):
    X_scaled_flat, Y_labels_raw, N_features, scaler_obj, feature_list_obj = preprocess_and_scale_data(file_path)

    df_temp = pd.DataFrame(X_scaled_flat)
    df_temp[USER_ID_COLUMN] = Y_labels_raw
    
    X_sequences_segmented = []
    Y_labels_segmented = []

    for user_name, group_df in df_temp.groupby(USER_ID_COLUMN):
        user_data_matrix = group_df.drop(USER_ID_COLUMN, axis=1).values.astype(np.float32)
        
        segmented_sequences, segmented_labels = create_short_sequences(user_data_matrix, user_name)
        
        X_sequences_segmented.extend(segmented_sequences)
        Y_labels_segmented.extend(segmented_labels)
        
    print(f"Total number of training samples created: {len(X_sequences_segmented)}")
    
    if not X_sequences_segmented:
        raise ValueError(f"No segments were generated. Data is too short for {TARGET_SEQUENCE_LENGTH} keystroke windows.")

    #Label encoding
    unique_labels = sorted(list(set(Y_labels_segmented)))
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)
    Y_labels_encoded = label_encoder.transform(Y_labels_segmented)

    #Padding the last datas
    X_padded = pad_sequences(
        X_sequences_segmented, 
        maxlen=TARGET_SEQUENCE_LENGTH, 
        dtype='float32', 
        padding='post', 
        truncating='post', 
        value=0.0
    )
    
    M_max = TARGET_SEQUENCE_LENGTH
    
    return X_padded, Y_labels_encoded, label_encoder, M_max, N_features, scaler_obj, feature_list_obj


def create_typenet_model(input_features, lstm_units=LSTM_UNITS, dropout_rate=0.5):
    """
    Implementing the Typenet triplet network from the thesis and github
    """
    input_tensor = Input(shape=(None, input_features), name='keystroke_input')
    x = Masking(mask_value=0.0, name='masking_layer')(input_tensor)
    x = BatchNormalization(name='bn_layer_1')(x)
    x = LSTM(units=lstm_units, return_sequences=True, name='lstm_layer_1')(x)
    x = Dropout(rate=dropout_rate, name='dropout_layer')(x)
    x = LSTM(units=lstm_units, return_sequences=False, name='lstm_layer_2')(x)
    output_tensor = BatchNormalization(name='bn_layer_2_embedding_f_x')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor, name='TypeNet_Feature_Extractor')
    return model

def create_softmax_training_model(feature_extractor, num_classes):
    """Creates the full model for Softmax Loss training."""
    f_x = feature_extractor.output
    classification_output = Dense(
        units=num_classes, 
        activation='softmax', 
        name='softmax_classification_head'
    )(f_x)
    full_model = Model(
        inputs=feature_extractor.input, 
        outputs=classification_output, 
        name='TypeNet_Softmax_Classifier'
    )
    full_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return full_model


def resave_model():   
    OLD_MODEL_PATH = FEATURE_MODEL_PATH
    NEW_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Trained_Model', 'Baseline', 'typenet_feature_extractor_tf'))  
    
    feature_list = joblib.load(FEATURE_LIST_PATH)
    N_features = len(feature_list)
    
    model = get_typenet_model_exact(
        input_shape=(TARGET_SEQUENCE_LENGTH, N_features)
    )

    model.compile(optimizer='adam', loss='mse') 
    model.load_weights(OLD_MODEL_PATH) 
    model.export(NEW_MODEL_PATH)


def get_typenet_model_exact(input_shape, lstm_units=128, dropout_rate=0.5):
    """Reconstruction of the TypeNet Feature Extractor."""
    input_tensor = Input(shape=input_shape, name='keystroke_input')
    x = Masking(mask_value=0.0, name='masking_layer')(input_tensor)
    x = BatchNormalization(name='bn_layer_1')(x)
    x = LSTM(units=lstm_units, return_sequences=True, name='lstm_layer_1')(x)
    x = Dropout(rate=dropout_rate, name='dropout_layer')(x)
    x = LSTM(units=lstm_units, return_sequences=False, name='lstm_layer_2')(x)
    output_tensor = BatchNormalization(name='bn_layer_2_embedding_f_x')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor, name='TypeNet_Feature_Extractor')
    return model


if __name__ == '__main__':
    
    model_size_mb = 0.0
    train_duration = 0.0

    try:
        X_data, Y_data, label_encoder, M_max, N_features, scaler_obj, feature_list_obj = \
            read_and_prepare_keystroke_data_full(file_path=PROCESSED_FILE_NAME)
        
        num_users = len(label_encoder.classes_)
        
        #Save the scaler and feature list
        joblib.dump(scaler_obj, SCALER_PATH)
        joblib.dump(feature_list_obj, FEATURE_LIST_PATH)
        
        #Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_data, Y_data, test_size=0.2, random_state=42, stratify=Y_data
        )

        print(f"\nTraining Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}, Classes: {num_users}")

        #Create model based on the thesis
        typenet_f_x = create_typenet_model(input_features=N_features)
        softmax_model = create_softmax_training_model(feature_extractor=typenet_f_x, num_classes=num_users)
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=PATIENCE_VALUE,
            restore_best_weights=True,
            verbose=1
        )
        
        # Start Timer
        start_time = time.time() 
        
        history = softmax_model.fit(
            x=X_train,
            y=Y_train,
            batch_size=64, 
            epochs=100,
            verbose=1,
            validation_split=0.2, 
            callbacks=[early_stopping]
        )
        
        # Stop Timer
        train_duration = time.time() - start_time 
        print(f"\nTraining Complete (Time: {train_duration:.2f} seconds)")
        
        #Save the extracted features
        typenet_f_x.save(FEATURE_MODEL_PATH)
        
        # Calculate model size
        if os.path.exists(FEATURE_MODEL_PATH):
            model_size_bytes = os.path.getsize(FEATURE_MODEL_PATH)
            model_size_mb = model_size_bytes / (1024 * 1024)
            
        #Test
        if X_test.shape[0] > 0:
            sample_input = X_test[0] 
            sample_input_reshaped = np.expand_dims(sample_input, axis=0) 
            embedding_vector = typenet_f_x.predict(sample_input_reshaped, verbose=0) 

        y_proba_raw = softmax_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_proba_raw, axis=1)
        target_names = label_encoder.classes_
        
        #Generate Report
        report_dict = classification_report(Y_test, y_pred, zero_division=0, target_names=target_names, output_dict=True)
        weighted_avg = report_dict['weighted avg']
        
        model_name = "TypeNet Softmax Classifier"
        precision = weighted_avg['precision']
        recall = weighted_avg['recall']
        f1_score = weighted_avg['f1-score']
        support = weighted_avg['support']
        
        print("\n" + "="*95)
        print("Model Performance Summary")
        print("="*95)
        
        # Print Header
        print(f"| {'Model':<30} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} | {'Support':^8} | {'Train-time (s)':^14} | {'Model Size (MB)':^15} |")
        # Print Separator
        print(f"|{'-'*30}-|{'-'*10}-|{'-'*10}-|{'-'*10}-|{'-'*8}-|{'-'*14}-|{'-'*15}-|")
        # Print Data Row
        print(f"| {model_name:<30} | {precision:^10.4f} | {recall:^10.4f} | {f1_score:^10.4f} | {support:^8} | {train_duration:^14.2f} | {model_size_mb:^15.2f} |")

        #Evaluation
        if num_users == 2:
            # Binary Assumption
            y_proba = y_proba_raw[:, 1]
            eer_far, roc_auc, y_proba_defined = np.nan, np.nan, True
            
            # ROC and EER Calculation
            fpr, tpr, thresholds = roc_curve(Y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            eer_index = np.argmin(np.abs(fpr + tpr - 1)) 
            eer_far = fpr[eer_index]
            eer_frr = 1 - tpr[eer_index] 
        else:
            eer_far = np.nan
            y_proba = None
            y_proba_defined = False

        # Accuracy
        accuracy = accuracy_score(Y_test, y_pred)
        print(f"Pontosság (Test): {accuracy:.4f}")
        if not np.isnan(eer_far):
            print(f"Equal Error Rate (EER): {eer_far:.4f}")
            
        # Classification report
        print("\nClassification Report:")
        print(classification_report(Y_test, y_pred, zero_division=0, target_names=target_names))

        # Confusion matrix
        cm = confusion_matrix(Y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title('Konfúziós Mátrix')
        plt.xlabel('Model által tippelt címke')
        plt.ylabel('Eredeti címke')
        plt.show()

        # ROC plot
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC görbe (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Véletlen (Random)') 
        plt.plot(eer_far, 1 - eer_frr, marker='o', markersize=7, color='red', label=f'EER pont ({eer_far:.4f})')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('FAR (False Acceptance Rate)')
        plt.ylabel('1 - FRR (True Accept Rate / Genuine Accept Rate)')
        plt.title(f'ROC (Receiver Operating Characteristic) Görbe (EER: {eer_far:.4f})')
        plt.legend(loc="lower right"); plt.grid(True)
        plt.show()
        print(f"\nArea Under the Curve (AUC): {roc_auc:.4f}")
        
        #Tensorflow/Keras bugfix for not loading the model correctly
        resave_model()

    except FileNotFoundError:
        print(f"\nFile '{PROCESSED_FILE_NAME}' not found. Please ensure '{PROCESSED_FILE_NAME}' is available.")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError in data processing: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during model execution: {e}")
        sys.exit(1)
