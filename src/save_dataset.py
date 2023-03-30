
import numpy as np
from  timeseries_classfication import diVibes
import os
from datetime import datetime
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Save dataset in numpy array')
    # Basic parameters
    parser.add_argument('--data_dir', type=str, default='data', help='The path of the training and testing dataset.')
    parser.add_argument('--output_dir', type=str, default='output', help='The path of the output directory.')    
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    trainer = diVibes()

    output_dir = os.path.join(args.output_dir, datetime.strftime(datetime.now(), '%m%d-%H%M%S'))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_dataset_path = os.path.join(args.data_dir, 'training')

    class_names = trainer.get_dataset_info(training_dataset_path)
    print('Classes in training dataset')
    print(class_names)

    with open(os.path.join(output_dir, 'class_name.txt'), 'w') as f:
        f.write(' '.join(class_names))

    # Load training dataset
    training_dataset, training_labels = trainer.get_dataset(training_dataset_path)

    print('Size of training dataset')
    print(training_dataset.shape)
    training_labels = np.array(training_labels)
    print(training_labels.shape)


    with open(os.path.join(output_dir, 'training_dataset.npy'), 'wb') as f:
        np.save(f, training_dataset)

    with open(os.path.join(output_dir, 'training_labels.npy'), 'wb') as f:
        np.save(f, training_labels)    

    # Load test dataset
    test_dataset_path = os.path.join(args.data_dir, 'test')
    test_dataset, test_labels = trainer.get_dataset(test_dataset_path)

    print('Size of test dataset')
    print(test_dataset.shape)
    test_labels = np.array(test_labels)
    print(test_labels.shape)

    with open(os.path.join(output_dir, 'test_dataset.npy'), 'wb') as f:
        np.save(f, test_dataset)

    with open(os.path.join(output_dir, 'test_labels.npy'), 'wb') as f:
        np.save(f, test_labels)   

