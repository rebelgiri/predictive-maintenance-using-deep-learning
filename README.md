# **Predictive Maintainance using Deep Learning**




## **Introduction**
In the era of Industry 4.0, sensors are installed on machinery for real-time data collection, processing, and automation. The collected data availed to build deep learning models for predictive maintenance and condition monitoring. One such sensor is the vibration sensor. They are used to detect several abnormalities in industrial machinery. In this task, a test setup is constructed, which simulates an industrial machine, and the data is collected using a vibration sensor installed upon it. Further, the collected data processed and benefited to build deep learning models and conduct investigations associated with the estimation of speed of rotors and rotating unbalance.

## **Architecture of the Classification Model**

![](https://github.com/rebelgiri/predictive-maintenance-using-deep-learning/blob/main/figures/FCN.png "Architecture of the model using FCN for time series classification")
<p align="center">
      <em>Architecture of the model using FCN for time series classification.</em>
</p>

## **Preprocessing Time Series Data using StandardScalar**

The time series data collected from the vibration sensor is 3D. Initially, the data is preprocessed using the StandardScaler method1 and fed to the model for training. The StandardScaler standardizes features by subtracting the mean and scaling to unit variance. The standard score z of a sample $x$ is calculated using equation below, where $μ$ is the mean and $s$ is the standard deviation.

$z = (x - u) / s$

<p align="center">
    <img src="https://github.com/rebelgiri/predictive-maintenance-using-deep-learning/blob/main/figures/100_x-direction_samples.png" align="center">
</p>


<p align="center">
      <em>Standardized time series X-direction vibration data.</em>
</p>

## **Usage**

Below arguements are passed while running the below scripts.
### *main.py*

This script is used to start training the model.

    1. --model_name: The name of the model.
    2. --data_dir: The path of the training and testing dataset.
    3. --output_dir: The path of the output directory.
    4. --N: The length of the time series slice. we use N to slice the complete time series data and stack them above each other.For example, the time series is of length 1000, and if N is 300, then three sliced time series data of length 300 are stacked one after the other with the same label.

<!-- ### save_dataset.py 

 This script is used to save datasets in numpy files.

    1. --data_dir : The path of the training and testing dataset.
    2. --output_dir: The path of the output directory. -->

### *load_model.py*

This script is used to load saved models for testing.
    
    1. --model_path: The path of the saved model.
    2. --test_path: The path of the test dataset.
    3. --output_dir: The path of the output directory.

## **Experiments**

### Estimation of the Rotating Unbalance

<p align="center">
    <img src="https://github.com/rebelgiri/predictive-maintenance-using-deep-learning/blob/main/figures/trainingAndValidationLoss.png" align="center" width="49%">
    <img src="https://github.com/rebelgiri/predictive-maintenance-using-deep-learning/blob/main/figures/trainingAndValidationAccuracy.png" align="center" width="49%">
</p>


<p align="center">
      <em>Training and validation learning curves while training Rotating Unbalance
Classifier.</em>
</p>


<p align="center">
    <img src="https://github.com/rebelgiri/predictive-maintenance-using-deep-learning/blob/main/figures/confusionMatrix.png" align="center" width="60%">
</p>


<p align="center">
      <em>Illustration of the performance of the Rotating Unbalance Classifier on the test
dataset using a confusion matrix.</em>
</p>



### Estimation of the Speed



<p align="center">
      <img src="https://github.com/rebelgiri/predictive-maintenance-using-deep-learning/blob/main/figures/speedClassifierTrainingAndValidationLoss.png" align="center" width="49%">
      <img src="https://github.com/rebelgiri/predictive-maintenance-using-deep-learning/blob/main/figures/speedClassifierTrainingAndValidationAccuracy.png" align="center" width="49%">
</p>

<p align="center">
      <em>Training and validation learning curves while training Speed
Classifier.</em>
</p>



<p align="center">
    <img src="https://github.com/rebelgiri/predictive-maintenance-using-deep-learning/blob/main/figures/speedClassifierConfusionMatrix.png" align="center" width="60%">
</p>


<p align="center">
      <em>Illustration of the performance of the Speed Classifier on the test dataset using a
confusion matrix.</em>
</p>

## **Conclusion**

The primary goal of this task is to draft a proof-of-concept for predictive maintenance
and condition monitoring of industrial machines using deep learning models. Therefore,
a test setup simulating an industrial machine is constructed. Later, classification models
are trained using the time series data collected from the vibration sensor installed upon
it. The investigations reveal that the trained Rotating Unbalance Classifier and Speed
Classifier are almost 100% accurate on unseen test data.
