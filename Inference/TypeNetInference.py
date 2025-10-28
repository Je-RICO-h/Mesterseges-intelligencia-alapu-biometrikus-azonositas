import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model 

TARGET_SEQUENCE_LENGTH = 5
PADDING_VALUE = 0.0

class TypeNetInference:
    def __init__(self, model_path, scaler_path, feature_list_path):
        
        #Load the model 
        self.model = TFSMLayer(
            model_path, 
            call_endpoint='serve', 
            name='Inference_Feature_Extractor_Layer'
        )
        
        #Load the Scaler and Feature List
        self.scaler = joblib.load(scaler_path)
        self.feature_list = joblib.load(feature_list_path)
        
        N_FEATURES = len(self.feature_list)
        
        INPUT_SEQUENCE_LENGTH = TARGET_SEQUENCE_LENGTH
        input_tensor = tf.keras.Input(shape=(INPUT_SEQUENCE_LENGTH, N_FEATURES), name='inference_input_tensor')
        output_tensor = self.model(input_tensor) 

        self.model = Model(inputs=input_tensor, outputs=output_tensor)
        self.input_features = self.model.input_shape[-1] 
        self.embedding_size = self.model.output_shape[-1]
        
        
    def __prepare_single_sequence(self, df_input):
        """
        Preprocess the data
        """
        df = df_input.copy()
        
        #Drop unnecessary columns
        df = df.drop(columns=['start_time_timestamp', 'end_time_timestamp', 'label', 'language'], errors='ignore')
        
        #Type conversion
        df['error'] = df['error'].astype(int)
        df['is_burst'] = df['is_burst'].astype(int)

        #OHE
        df_encoded = pd.get_dummies(df, columns=['key', 'combination'], dtype=int)

        #Interpollation of missing data
        for col in ['end_time', 'variation_in_typing_speed', 'hold_time', 'interval']:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].interpolate(method='linear', limit_direction='both').fillna(PADDING_VALUE)

        #Reindexing
        df_final = df_encoded.reindex(columns=self.feature_list, fill_value=0)
        
        #Minmax scaler
        data_scaled = self.scaler.transform(df_final.values)
        
        if data_scaled.shape[0] < TARGET_SEQUENCE_LENGTH:
            raise ValueError(f"Input data has only {data_scaled.shape[0]} keystrokes. Requires at least {TARGET_SEQUENCE_LENGTH}.")
        
        segment = data_scaled[-TARGET_SEQUENCE_LENGTH:, :]
        
        return np.expand_dims(segment, axis=0)

    def predict_embedding(self, raw_data_df):
        """
        Inference predict
        """
        try:
            prepared_tensor = self.__prepare_single_sequence(raw_data_df)
            
            embedding = self.model.predict(prepared_tensor, verbose=0)
            
            print('\n')
            return embedding
        
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None