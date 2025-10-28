from Inference.TypeNetInference import TypeNetInference as Inference
from collections import Counter
from pynput.keyboard import Key, Listener, KeyCode
from Logging.LoggingUtils import save_csv, get_keyboard_language
from sys import exit
import time
import datetime
import pandas as pd 
import numpy as np

# Global constant from TypeNet
TARGET_SEQUENCE_LENGTH = 5

timeout = 10 #Seconds until we decide that we are not at the keyboard

class KeyLogger:

    def __init__(self, label, file_path, threshold=50, model_paths={}):
        self.file_path = file_path #Where to save the file
        self.label = label # Who does the logging
        self.pressed_keys = set() #Aux set to hold the pressed_keys
        self.data = [] # Variable to hold the logging data temporarily
        self.total_start_time = 0 #Variable to hold the start of the logging
        self.key_counter = 0 # Total pressed keys
        self.key_combinations = Counter() #Count the frequences of combination
        self.threshold = threshold # How frequently we should save
        self.error_counter = 0 # Total errors
        self.sequence_length = threshold # Sequence threshold where we begin a new file
        self.inference_mode = False #Inference or not
        
        self.user_template = np.zeros((1, 128)) + 0.5 # A reproducible vector of 0.5
        self.VERIFICATION_THRESHOLD = 0.73
        
        #Inference Mode
        if len(model_paths) > 0:
            self.inference_mode = True
            
            if all(k in model_paths for k in ['model_path', 'scaler_path', 'feature_list_path']):
                self.inference_model = Inference(
                    model_paths['model_path'], 
                    model_paths['scaler_path'], 
                    model_paths['feature_list_path']
                )
            else:
                raise Exception(f"Can't start inference. Check the model paths")
        
        
    def new_state(self):
        '''
        Function to reset the state when we start a new file (Reset counters, time etc.)
        '''
        self.key_counter = 0 
        self.total_start_time = time.time() 
        self.pressed_keys = set() 
        self.data = [] 
        self.key_combinations = Counter() 
        self.error_counter = 0


    def calculate_kps(self, elapsed_time):
        """
        Function to calculate the current average key per second
        """
        return self.key_counter / elapsed_time if elapsed_time > 0 else 0


    def check_threshold(self):
        '''
        Function to check if we reached the threshold then save, in inference we predict
        '''
        if self.key_counter >= self.threshold:
            
            if self.inference_mode:
                try: 
                    start_data_proc_time = time.time()
                    
                    #Feature calculation
                    self.calculate_derived_features() 
                    raw_data_df = pd.DataFrame(self.data) 
                    
                    end_data_proc_time = time.time()
                    data_proc_time = end_data_proc_time - start_data_proc_time
                    
                    start_model_run_time = time.time()
                    
                    #Predict
                    embedding = self.inference_model.predict_embedding(raw_data_df)
                    
                    end_model_run_time = time.time()
                    model_run_time = end_model_run_time - start_model_run_time
                    
                    #Classification
                    if embedding is not None:
                        distance = np.linalg.norm(embedding - self.user_template) 
                        
                        if distance < self.VERIFICATION_THRESHOLD:
                            predicted_label = self.label
                            confidence = np.clip(1.0 - (distance / self.VERIFICATION_THRESHOLD), 0.0, 1.0)
                        else:
                            predicted_label = "Other"
                            confidence = np.clip((distance - self.VERIFICATION_THRESHOLD) / (3.0 * self.VERIFICATION_THRESHOLD), 0.0, 1.0)
                            
                        keystrokes_segment = [d['key'].replace("Key.", "") for d in self.data[-TARGET_SEQUENCE_LENGTH:]]
                        keystrokes_str = " -> ".join(keystrokes_segment)

                        print(f"Adatfeldolgozási idő: {data_proc_time:.4f} másodperc")
                        print(f"Jóslási idő (Modell futás): {model_run_time:.4f} másodperc")
                        print(f"Eredmény: {predicted_label} (Magabiztosság: {confidence:.2f})")
                        
                        self.new_state()
                    else:
                        self.new_state()
                        
                except Exception as e:
                    self.save_logging() 
                    self.new_state() 
                    
            else:
                self.save_logging()
                self.new_state()


    def reset_data(self):
        '''
        Function to reset the data, after keyboard inactivity or reaching threshold
        '''
        self.total_start_time = time.time()
        self.data = []
        self.key_counter = 0
        self.pressed_keys = set()
        self.key_combinations = Counter()
        self.error_counter = 0


    def calculate_derived_features(self):
        """
        Calculates derived features
        """
        if not self.data:
            return

        df = pd.DataFrame(self.data)
        
        #is_burst
        BURST_THRESHOLD = 0.050 # 50 milliseconds threshold
        df['is_burst'] = (df['interval'] <= BURST_THRESHOLD).astype(int)
        
        #variation_in_typing_speed
        window = 5
        df['variation_in_typing_speed'] = df['key_per_second'].rolling(window=window, min_periods=1).std().fillna(0.0)

        self.data = df.to_dict('records')

    def on_press_append_data(self, key, st, st_timestamp):
        """
        Function to prepare the data for save, we need this function as
        we have 2 different function for press and release actions
        :param key: Key pressed
        :param st: Start time of the key pressed
        :param st_timestamp The start timestamp
        """

        #Current time from the start
        st = st - self.total_start_time

        #New entry
        self.data.append({"key" : str(key),
                          "start_time" : st,
                          "key_per_second" : self.calculate_kps(st),
                          "combination" : None,
                          "error" : False,
                          "start_time_timestamp" : st_timestamp,
                          # Initialize fields that will be added later by on_release
                          "end_time": None, 
                          "hold_time": None,
                          "end_time_timestamp": None,
                          "language": None,
                          "seek_time": None, 
                          "variation_in_typing_speed": None, 
                          "interval": None 
                          })

        #If this is not the first entry then we can calculate the seek time, else this was the first key press
        if len(self.data) > 1:
            prev_data = self.data[-2]
            
            # Determine the end time of the previous event (used for flight/seek time)
            prev_end_time = prev_data.get('end_time', prev_data['start_time'])
            
            if prev_end_time is not None and (st - prev_end_time > timeout):
                # Inactivity detected
                self.reset_data()
                self.data[-1]['seek_time'] = 0 # Current key is the start of a new sequence
            else:
                self.data[-1]['seek_time'] = st - prev_end_time if prev_end_time is not None else 0
                
        else:
            self.data[-1]['seek_time'] = 0
            self.data[-1]['key_per_second'] = 0
            
        # Add error rate and accuracy (using current key_counter)
        self.data[-1]["error_rate"] = self.error_counter / self.key_counter if self.key_counter > 0 else 0
        self.data[-1]["accuracy"] = (self.key_counter - self.error_counter) / self.key_counter if self.key_counter > 0 else 1.0
        
        #If there are multiple keys pressed, then it's a combination, we save this combination, and increase its counter
        if len(self.pressed_keys) > 1:
            s = ""
            for c in sorted(list(self.pressed_keys)): 
                s += c + ","
            self.data[-1]['combination'] = s[:-1]
            self.key_combinations[s[:-1]] += 1

        #Check if the key that was pressed it was an error (Backspace handling)
        if key == Key.backspace:
            loc = -1
            # Find the first non-backspace and non-error key to mark as an error
            while abs(loc) < len(self.data) and str(self.data[loc]['key']) == str(Key.backspace):
                loc -= 1
                
            while abs(loc) < len(self.data) and str(self.data[loc]['error']) == str(True):
                loc -= 1
                
            # If a previous key was successfully found
            if abs(loc) < len(self.data):
                self.error_counter += 1
                self.data[loc]['error'] = True
                
                # Update error/accuracy metrics for the marked error entry
                self.data[loc]["error_rate"] = self.error_counter / self.key_counter
                self.data[loc]["accuracy"] = (self.key_counter - self.error_counter) / self.key_counter

    def on_release_append_data(self, key, et, ts, lang):
        index = -1

        #We search for the last not released key
        try:
            while self.data[index]['key'].upper() != key.upper():
                index -= 1
        except IndexError:
            return

        #If there are more than one same key next to each other, we find the first one not updated
        try:
            while 'end_time' not in self.data[index]:
                if index > -len(self.data) and self.data[index-1]['key'].upper() == key.upper() and 'end_time' not in self.data[index-1]:
                    index -= 1
                else:
                    break
        except IndexError:
            pass

        #Get the time since start of the logging
        et = et - self.total_start_time

        #Insert end_time and calculate hold_time
        self.data[index]['end_time'] = et
        self.data[index]['hold_time'] = et - self.data[index]['start_time']
        self.data[index]['end_time_timestamp'] = ts
        
        #We left the keyboard
        if et - self.data[index]['start_time'] < 0:
            self.data[index]['hold_time'] = 0
        
        self.data[index]['language'] = lang

        prev_end_time = None 

        # Check if a *previous* completed entry exists.
        prev_entry_index = index - 1 

        # Search logic remains: find the last entry with a completed 'end_time'
        while prev_entry_index >= -len(self.data) and 'end_time' not in self.data[prev_entry_index]:
            prev_entry_index -= 1
            
        #Define prev_end_time if a valid previous entry was found
        if prev_entry_index >= -len(self.data):
            prev_end_time = self.data[prev_entry_index]['end_time']
            
        #Perform the calculation
        if prev_end_time is not None:
            self.data[index]['interval'] = et - prev_end_time
        else:
            self.data[index]['interval'] = 0.0
            
    def on_press(self, key):
        if len(self.data) == 0:
            self.total_start_time = time.time() #Start of the logging
        
        start_time = time.time()

        if key == Key.esc:
            self.end_logging()

        if str(key).upper() not in self.pressed_keys:
            self.pressed_keys.add(str(key).upper())

        self.key_counter += 1
        self.on_press_append_data(key, start_time, start_time)


    def on_release(self, key):
        end_time = time.time()

        keyboard_language = get_keyboard_language() # Get the current keyboard language used

        #Bugfix for hungarian keyboard layout
        if keyboard_language == "Hungarian":
            if str(key) == "'z'": 
                key = KeyCode.from_char("z")
            elif str(key) == "'j'" and "'Í'" in self.pressed_keys:
                key = KeyCode.from_char("í")

        self.pressed_keys.discard(str(key).upper())

        #Update the other part of the data
        self.on_release_append_data(str(key), end_time, end_time, keyboard_language)
        
        self.check_threshold()


    def start_logging(self):
        """
        Initializes the text, the parameters, and starts the logging
        """
        if self.inference_mode:
            print(f"{datetime.datetime.now()} - TypeNet Inference started! (Threshold: {self.threshold})")
        else:
            print(f"{datetime.datetime.now()} - logging started! (Threshold: {self.threshold})")
        
        #Start the listener
        with Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()

    def end_logging(self):
        """
        Function to safely end the logging
        """
        if not self.inference_mode:
            self.save_logging()
        
            print(f"{datetime.datetime.now()} - logging finished!")
        
        exit(0)

    def save_logging(self):
        """
        Function to save the collected logging data
        """
        
        try:
            completed_data = [d for d in self.data if d.get('end_time') is not None]
            save_csv(self.file_path, self.label, completed_data)
            
        except IndexError:
            return
            
        print(f"{datetime.datetime.now()} - logging state saved!")