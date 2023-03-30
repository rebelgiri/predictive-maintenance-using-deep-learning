

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from timeseries_classfication import diVibes
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import argparse
import glob
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Predict')
    # Basic parameters
    parser.add_argument('--model_path', type=str,
                        default='data', help='The path of the saved model.')
    parser.add_argument('--test_path', type=str, default='test',
                        help='The path of the test dataset.')
    parser.add_argument('--output_dir', type=str,
                        default='output', help='The path of the test dataset.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    tester = diVibes()

    model = keras.models.load_model(args.model_path)
    output_dir = args.output_dir + '_model_prediction_results'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    class_names = ['A2', 'S1', 'A1', 'A1A2']

    dataset = np.array([])

    for file in sorted(glob.glob(args.test_path + '/*.txt')):

        #print('Loaded measurements from the file ' + file.split('/')[-1])

        timeseries = np.loadtxt(file, delimiter=';', dtype=float,
                                skiprows=29, usecols=(1, 2, 3), encoding='latin2')

        # Standardize the timeseries data
        timeseries = tester._data_standardization(timeseries)

        dataset = np.vstack([dataset, timeseries[0:tester.N].reshape(1, tester.N, 3)])\
            if dataset.size else timeseries[0:tester.N].reshape(1, tester.N, 3)

        # dataset = np.vstack([dataset,
        #                      timeseries[tester.N:tester.N + tester.N].reshape(1, tester.N, 3)])
        # dataset = np.vstack([dataset,
        #                      timeseries[tester.N + tester.N: len(timeseries) - 1].reshape(1, tester.N, 3)])

    test_pred = np.argmax(model.predict(dataset), axis=1)
        
    for i, file in enumerate(sorted(glob.glob(args.test_path + '/*.txt'))):
        print('The time series data from the file ' + file.split('/')[-1] + ' predicted as {}'.format(
            class_names[test_pred[i]]))

    # test_loss, test_acc = model.evaluate(dataset, np.array(true_labels))
    # print('Test accuracy = {} and loss = {}'.format(test_acc, test_loss))

    # cm = confusion_matrix(np.array(true_labels), test_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    # plt.close()

    # for i, file in enumerate(sorted(glob.glob(args.test_path + '/*.txt'))):
    #     print('The file ' + file.split('/')[-1] + ' of class {} -> but predicted as {}'.format(
    #         class_names[true_labels[i]], class_names[test_pred[i]]))
