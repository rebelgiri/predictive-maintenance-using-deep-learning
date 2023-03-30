"""
Title: Timeseries classification
Author: Giriraj Pawar   
Date created: 27.01.2023
Last modified: 27.01.2023
Description: Training a timeseries classifier on the Vibration dataset.
Accelerator: GPU
"""

from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob
from tqdm import tqdm

class diVibes:

    def __init__(self, len_series=300001, N=100000):
        self.len_series = len_series
        self.N = N

    def _data_standardization(self, timeseries):
        scaler = StandardScaler()
        return scaler.fit_transform(timeseries)

    def get_dataset_info(self, dataset_path):
        # retuns the folder names found under the training dataset
        class_names = []
        for folder in glob.glob(dataset_path + '/*'):
            class_names.append(folder.split('/')[-1])
        return class_names


    def get_dataset(self, dataset_path, class_names):
        """
        Return a Standardized dataset.
        """
        dataset = np.array([])
        labels = []

        for folder in glob.glob(dataset_path + '/*'):
            i = class_names.index(folder.split('/')[-1])
            j = 0
            for file in glob.glob(folder + '/*.txt')[0:150]:
                timeseries = np.loadtxt(file, delimiter=';', dtype=float,
                                        skiprows=29, usecols=(1, 2, 3), encoding='latin2') 
                
                # Standardize the timeseries data
                timeseries = self._data_standardization(timeseries)

                # Here we use N to slice the complete time series data and stack them above each other. 
                # For example, the time series is of length 1000, and if N is 300, then three sliced time series data of length 300 
                # are stacked one after the other with the same label.
                dataset = np.vstack([dataset, timeseries[0:self.N].reshape(1, self.N, 3)])\
                    if dataset.size else timeseries[0:self.N].reshape(1, self.N, 3)
                dataset = np.vstack([dataset, timeseries[self.N:self.N + self.N].reshape(1, self.N, 3)])
                dataset = np.vstack([dataset, timeseries[self.N + self.N: len(timeseries) - 1].reshape(1, self.N, 3)])
        
                labels.extend([i, i, i])
                j = j + 1
            print('Loaded ' + str(j) + ' measurements of class ' + folder.split('/')[-1])

        return dataset, labels


    def get_model(self, input_shape, num_classes):
        """
        Return a FCN model. It is a Fully Convolutional Neural Network originally proposed in
        [this paper](https://arxiv.org/abs/1611.06455).
        """
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)
        #conv1 = keras.layers.Dropout(0.2)(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)
        #conv2 = keras.layers.Dropout(0.2)(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)
        

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)


