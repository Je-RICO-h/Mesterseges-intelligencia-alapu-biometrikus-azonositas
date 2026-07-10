from Logging.Keyboard_Logger_Baseline import KeyLogger
import time
from sys import exit

file_path = "Data" # The path of the file to save the logging
threshold = 5

model_path = {
    'model_path':'Trained_Model/Baseline/typenet_feature_extractor_tf',
    'scaler_path': 'Trained_Model/Baseline/typenet_minmax_scaler.pkl',
    'feature_list_path': 'Trained_Model/Baseline/typenet_feature_list.pkl'
}

#Inference mode
inference_mode = True

def main_menu():
    print("-"*25)
    label = input("Enter Label name: ")
    print("-"*25)
    
    if not inference_mode:
        input("Press enter to start the logging")
        training = KeyLogger(label, file_path, threshold=threshold)
    else:
        input("Press enter to start the inference")
        training = KeyLogger(label, file_path, threshold=threshold, model_paths=model_path)
    
    #Wait for pressed enter to clear (The release event catches it)
    time.sleep(0.5)

    try:
        training.start_logging()
    except KeyboardInterrupt:
        training.end_logging()
        exit(0)

def main():
    main_menu()

if __name__ == "__main__":
    main()