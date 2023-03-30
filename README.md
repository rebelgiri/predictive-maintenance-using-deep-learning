# Predictive Maintainance using Deep Learning




## Introduction
In the era of Industry 4.0, sensors are installed on machinery for real-time data collection, processing, and automation. The collected data availed to build deep learning models for predictive maintenance and condition monitoring. One such sensor is the vibration sensor. They are used to detect several abnormalities in industrial machinery. In this task, a test setup is constructed, which simulates an industrial machine, and the data is collected using a vibration sensor installed upon it. Further, the collected data processed and benefited to build deep learning models and conduct investigations associated with the estimation of speed of rotors and rotating unbalance.

## Usage

Below arguements are passed while running scripts.
### main.py 

This script is used to start training the model.

    1. --model_name: The name of the model.
    2. --data_dir: The path of the training and testing dataset.
    3. --output_dir: The path of the output directory.
    4. --N: The length of the time series slice. we use N to slice the complete time series data and stack them above each other.For example, the time series is of length 1000, and if N is 300, then three sliced time series data of length 300 are stacked one after the other with the same label.

### save_dataset.py 

This script is used to save datasets in numpy files.

    1. --data_dir : The path of the training and testing dataset.
    2. --output_dir: The path of the output directory.

### load_model.py 

This script is used to load saved models.
    
    1. --model_path: The path of the saved model.
    2. --test_path: The path of the test dataset.
    3. --output_dir: The path of the output directory.
    


