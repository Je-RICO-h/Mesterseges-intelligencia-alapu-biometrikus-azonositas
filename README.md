# Accelerated machine learning based biometric identification
## Authors: Pál Erik, Lakatos Róbert, Prof. Dr. Hajdu András

![presentation](https://github.com/user-attachments/assets/3f5c2387-5fbe-49e4-8374-6c95eb1d1dfe)


In the digital age, protecting data assets has become one of the most critical security challenges. This task is further complicated by user demands for convenience, which often lead to insufficient protection. Traditional authentication procedures requiring user intervention, such as passwords or two-factor authentication are no longer capable of adequately managing the growing risk of unauthorized access. Consequently, it has become essential to implement passive and continuous identification solutions that can confirm the user's presence with high accuracy.

In our paper, we present an XGBoost-based machine learning method that infers identity solely from the user's typing dynamics. The proposed approach aims for the continuous identification of the person in front of the computer through the analysis of low-dimensional features, such as keystroke hold times, flight times, typing speed, and error rates. We compared our solution with the current State of the art (SOTA) Long Short-Term Memory (LSTM) deep learning model [1] [2]. Our developed model achieved 99% accuracy on the benchmark dataset, representing a 1.2% improvement over the LSTM-based method.
Furthermore, the low resource requirements of the XGBoost model allow for local, real-time execution: on Intel i7-7700K and i3-10100 processors, we achieved a response time of 63 ms with only 5% CPU load. In this paper, we formalized typing dynamics as a time-dependent stochastic process, providing a statistical foundation for the analysis. Additionally, through spectral analysis, we formalized the time-rhythm components characteristic of the user.

We have summarized the most important parameters in the table below, supplemented with training and model size characteristics.

<img width="640" height="451" alt="2" src="https://github.com/user-attachments/assets/29d0d26b-efcb-41b1-8e1d-aab858b456ea" />


In addition to personal computer execution, our implementation is compatible with the industrial NVIDIA Morpheus SDK framework, which enhances the system's flexibility and integrability. Consequently, our solution simultaneously ensures privacy and seamless background operation without sacrificing user convenience.

The proposed technology enables the introduction of a new, proactive security layer that can either replace or reinforce traditional authentication methods. This approach paves the way for future autonomous security systems capable of executing immediate defensive measures—such as locking the device—upon suspicion of unauthorized intervention.

**Our solution is open-source. To ensure scientific reproducibility and proper attribution, any use or further development of the work is subject to appropriate citation (reference).**

Further information, additional figures, and detailed results can be found in our study.

**References**

[1] BiDAlab, “TypeNet GitHub Repository,” [Online]. Available: https://github.com/BiDAlab/TypeNet. [Accessed: 20 October 2025].

[2] A. Alejandro, M. Aythami, M. J. V., V.-R. Ruben, and F. Julian, “TypeNet: Deep Learning Keystroke Biometrics,” IEEE Transactions on Biometrics, Behavior, and Identity Science, 2021.

# Working of the program

To facilitate testing, we provide the dataset used in our study and describe the key components of the implementation below:

### Data
Contains the datasets collected and prepared by us for testing. Newly gathered data is also saved here.

### Data_Processed
Contains the raw data processed by the .ipynb files found in the "Processing" folder. This data is already organized, augmented, and supplemented with all calculated features.

### Inference
A collection of model loader and prediction classes and functions, categorized by model type.

### Logging
Contains the custom-developed keylogger files and code. While categorized by model, we recommend using the Keyboard_Logger_XgBoost.py script for data collection.

### Model_Training
Programs used for training and implementing the models, grouped by model type.

### Processing
Contains procedures for augmenting and processing raw data; outputs are directed to the Data_Processed folder.

### Trained_Model
Contains pre-trained models ready for inference along with necessary auxiliary files (e.g., OHE features, Label Encoders), organized by model type.


### Main files
The program is managed via the two Train_data_collect_(Model).py files, depending on which model you intend to use for inference.
- model_path: This variable contains the paths to the models; by default, no modification is required.
- threshold: Use this to adjust the required number of collected samples.
- file_path: This parameter allows you to change the save directory for the data.
- inference_mode:
  - True: The program runs in inference mode using the appropriate loaded model.
  - False: The program runs in data collection mode.
- Running the program: Simply provide the desired label name when prompted, press Enter to start, and begin typing. The program handles all other tasks automatically in the background.

### Additional Information
- The libraries and Conda environment required to run the system can be found in the environment.yml and Requirements.txt files.

# Usage and Citation
We welcome the use of our code and methodology in scientific research, commercial projects, or other open-source initiatives. To maintain the integrity of our work, please ensure that any use of code, data, or results from this project includes a proper reference to the original source according to the proper format.
