# diVibes

## Name

Digital 3D Broadband Vibration Sensors for Improved Machine Monitoring
using Machine Learning. In short we call this project diVibes.


## Description
The short name of the project is diVibes. The repo is used to conduct a series of experiments on the time series data collected from the diVibes test setup. Each script takes different arguments. For now these scripts are not generic, as per the future requirements scripts will be modified and README is updated. For now, let's describe the usage of each script and their arguments.

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
    


