# Predictive Maintainance Using Deep Learning

## Name




## Description
The repo is used to conduct a series of experiments on the time series data collected from the test setup. Each script takes different arguments. For now these scripts are not generic, as per the future requirements scripts will be modified and README is updated. For now, let's describe the usage of each script and their arguments.

## Usage

### main.py 
    - *--model_name*: The name of the model.
    - *--data_dir*: The path of the training and testing dataset.
    - *--output_dir*: The path of the output directory.
    - *--N*: The length of the time series slice. we use N to slice the complete time series data and stack them above each other.For example, the time series is of length 1000, and if N is 300, then three sliced time series data of length 300 are stacked one after the other with the same label.
    - *--use_saved_data*: Set to 1 if you want to load the saved training and test dataset in the numpy file. The dataset can be saved using the script save_dataset.py, otherwise 0.

### save_dataset.py 

This script is used to save datasets in numpy files.

    - *--data_dir* : The path of the training and testing dataset.
    - *--output_dir*: The path of the output directory.

### load_model.py 

This script is used to load saved models.
    
    - *--model_path*: The path of the saved model.
    - *--test_path*: The path of the test dataset.
    - *--output_dir*: The path of the output directory.
    


