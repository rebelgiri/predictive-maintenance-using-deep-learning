

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from  timeseries_classfication import diVibes
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import argparse
import glob
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Predict')
    # Basic parameters
    parser.add_argument('--model_path', type=str, default='data', help='The path of the saved model.')
    parser.add_argument('--test_path', type=str, default='test', help='The path of the test dataset.')
    parser.add_argument('--output_dir', type=str, default='output', help='The path of the test dataset.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    tester = diVibes()
    
    model = keras.models.load_model(args.model_path)
    output_dir = args.output_dir + '_model_prediction_results'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    class_names = ['A1_1000', 'A1_1200', 'A1_600', 'A1_8000']


    # class_names = ['A2', 'S1', 'A1', 'A1_A2']
    # true_labels = ['S1', 'S1', 'S1',  
    #             'A1_A2', 'A1_A2', 'A1_A2',
    #             'A2', 'A2', 'A2', 
    #             'A1_A2', 'A1_A2', 'A1_A2',
    #             'A1_A2', 'A1_A2', 'A1_A2',
    #             'A1', 'A1', 'A1',
    #             'S1', 'S1', 'S1']

    # true_labels = [1, 1, 1,
    #             3, 3, 3,
    #             0, 0, 0,
    #             3, 3, 3,
    #             3, 3, 3,
    #             2, 2, 2,
    #             1, 1, 1]

    # Blind-dataset was collected at 1200 rpm.
    # B00.txt is of Class: S1
    # B01.txt is of Class: A2 -> A1_A2
    # B02.txt is of Class: A2
    # B03.txt is of Class: A1_A2 (this file was tricky, I merged to files: the first half from A1 and the second one from A2)
    # B05.txt is of Class: A2 -> A1_A2
    # B06.txt is of Class: A1_A2 -> A1
    # B07.txt is of Class: S1

    true_labels = [1, 1, 1,
                1, 1, 1,
                1, 1, 1,
                1, 1, 1,
                1, 1, 1,
                1, 1, 1,
                1, 1, 1]


    dataset = np.array([])
    for file in sorted(glob.glob(args.test_path + '/*.txt')):

        print('Loaded measurements from the file ' + file.split('/')[-1])
      
        timeseries = np.loadtxt(file, delimiter=';', dtype=float,
                                skiprows=29, usecols=(1, 2, 3), encoding='latin2') 
        
        # Standardize the timeseries data
        timeseries = tester._data_standardization(timeseries)

        dataset = np.vstack([dataset, timeseries[0:tester.N].reshape(1, tester.N, 3)])\
            if dataset.size else timeseries[0:tester.N].reshape(1, tester.N, 3)
    
        dataset = np.vstack([dataset, timeseries[tester.N:tester.N + tester.N].reshape(1, tester.N, 3)])
        
        dataset = np.vstack([dataset, timeseries[tester.N + tester.N: len(timeseries) - 1].reshape(1, tester.N, 3)])


    test_pred = np.argmax(model.predict(dataset), axis=1)


    test_loss, test_acc = model.evaluate(dataset, np.array(true_labels))
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)
        
    cm = confusion_matrix(np.array(true_labels), test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

        


        

