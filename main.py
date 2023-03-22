import tensorflow as tf
from tensorflow.keras import utils
import numpy as np
import matplotlib.pyplot as plt
from timeseries_classfication import diVibes
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import argparse
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow_datasets as tfds
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Configure the command-line arguments to train the model.')
    # Basic arguments
    parser.add_argument('--model_name', type=str,
                        default='FCN', help='The name of the model.')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='The path of the training and testing dataset.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The path of the output directory.')
    parser.add_argument('--N', type=int, default=100000,
                        help='The maximum length of sliced timeseries data.')
    args = parser.parse_args()
    return args


def data_standardization(timeseries):
    scaler = StandardScaler()
    return scaler.fit_transform(timeseries)


def slice_and_stack_timeseries(filepath, label, N):

    sliced_timeseries = np.array([])

    timeseries = np.loadtxt(filepath.numpy().decode(), delimiter=';', dtype=float,
                            skiprows=29, usecols=(1, 2, 3),
                            encoding='latin2')

    timeseries = data_standardization(timeseries)

    sliced_timeseries = timeseries[0:N].reshape(1, N, 3)

    sliced_timeseries = np.vstack(
        [sliced_timeseries, timeseries[N: N + N].reshape(1, N, 3)])

    sliced_timeseries = np.vstack(
        [sliced_timeseries, timeseries[N + N: len(timeseries) - 1].reshape(1, N, 3)])

    sliced_timeseries_tensor = tf.convert_to_tensor(
        sliced_timeseries, dtype=tf.float32)

    labels = tf.reshape(tf.concat([label, label, label], axis=0), [-1])

    return sliced_timeseries_tensor, labels


def get_label(file_path, class_names):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # Integer encode the label
    return parts[-2] == class_names


def filepath_label_ds(file_path, class_names):
    label = tf.where(get_label(file_path, class_names))
    return file_path, label


if __name__ == "__main__":

    args = parse_args()

    output_dir = os.path.join(
        args.output_dir, datetime.strftime(datetime.now(), '%m%d-%H%M%S'))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_dir = "logs/fit/"
    log_dir = os.path.join(output_dir, log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    training_dataset_path = os.path.join(args.data_dir, 'training')
    test_dataset_path = os.path.join(args.data_dir, 'test')

    # batch_size = 1
    # seed = 42

    # train_ds = tf.data.Dataset.list_files(str(pathlib.Path(training_dataset_path)/'*/*'),
    #                                       shuffle=False)
    # class_names = np.array([item.name for item in pathlib.Path(training_dataset_path).glob('*')])
    # print(class_names)
    # no_of_classes = len(class_names)
    # train_ds = train_ds.map(lambda x: filepath_label_ds(x, class_names))

    # train_ds = train_ds.map(lambda filepath, label: tf.py_function(slice_and_stack_timeseries,
    #                                                                [filepath, label, args.N], (tf.float32, tf.int64)))

    trainer = diVibes()
    class_names = trainer.get_dataset_info(training_dataset_path)
    print('Classes in training dataset {0}'.format(class_names))

    # load training dataset
    training_dataset, training_labels = trainer.get_dataset(
        training_dataset_path, class_names)

    # load test dataset
    test_dataset_path = os.path.join(args.data_dir, 'test')
    test_dataset, test_labels = trainer.get_dataset(
        test_dataset_path, class_names)

    print('Size of training dataset {}'.format(training_dataset.shape))
    training_labels = np.array(training_labels)
    print('Size of training dataset labels {}'.format(training_labels.shape))

    print('Size of test dataset {}'.format(test_dataset.shape))
    test_labels = np.array(test_labels)
    print('Size of test dataset labels {}'.format(test_labels.shape))



    # train_ds = tf.data.Dataset.from_tensor_slices((training_dataset, training_labels))
    # test_ds = tf.data.Dataset.from_tensor_slices((test_dataset, test_labels))

    # visualize the data. Here we visualize one timeseries example for each class in the dataset.
    classes = np.unique(training_labels, axis=0)

    plt.figure()
    for c in classes:
        c_x_train = training_dataset[training_labels == c]
        c_x_train = c_x_train.reshape(
            training_dataset[training_labels == c].shape[0], 3, training_dataset.shape[1])
        plt.plot(c_x_train[0][0][0:500], label="class " + str(c))
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, '500_x-direction_samples.png'))
    plt.close()

    plt.figure()
    for c in classes:
        c_x_train = training_dataset[training_labels == c]
        c_x_train = c_x_train.reshape(
            training_dataset[training_labels == c].shape[0], 3, training_dataset.shape[1])
        plt.plot(c_x_train[0][0][0:100], label="class " + str(c))
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, '100_x-direction_samples.png'))
    plt.close()

    plt.figure()
    for c in classes:
        c_x_train = training_dataset[training_labels == c]
        c_x_train = c_x_train.reshape(
            training_dataset[training_labels == c].shape[0], 3, training_dataset.shape[1])
        plt.plot(c_x_train[0][0][0:1000], label="class " + str(c))
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, '1000_x-direction_samples.png'))
    plt.close()

    """
    Finally, in order to use `sparse_categorical_crossentropy`, we will have to count
    the number of classes beforehand.
    """
    num_classes = len(np.unique(training_labels))

    # Create a model
    model = trainer.get_model(training_dataset.shape[1:], num_classes)

    # Train the model
    model_name = args.model_name + '_' + \
        datetime.strftime(datetime.now(), '%m.%d-%H.%M.%S') + '.h5'
    epochs = 300
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, model_name), save_best_only=True, monitor="val_loss", verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        tensorboard_callback,
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, verbose=1),
    ]

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(
        training_dataset,
        training_labels,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
        shuffle=True
    )

    # Evaluate model on test data
    model = keras.models.load_model(os.path.join(output_dir, model_name))
    test_pred = model.predict(test_dataset)

    cm = confusion_matrix(test_labels, np.argmax(test_pred, axis=1))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Plot the model's test accuracy and loss
    test_loss, test_acc = model.evaluate(test_dataset, test_labels)
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    # Plot the model's training and validation loss
    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("Training and Validation Sparse Categorical Accuracy")
    plt.ylabel('Sparse Categorical Accuracy', fontsize="large")
    plt.xlabel("Epochs", fontsize="large")
    plt.legend(["Train", "Val"], loc="best")
    # plt.show()
    plt.savefig(os.path.join(
        output_dir, 'training_and_validation_accuracy.png'))
    plt.close()

    metric = "loss"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("Training and Validation Sparse Categorical Loss")
    plt.ylabel('Loss', fontsize="large")
    plt.xlabel("Epochs", fontsize="large")
    plt.legend(["Train", "Val"], loc="best")
    # plt.show()
    plt.savefig(os.path.join(output_dir, 'training_and_validation_loss.png'))
    plt.close()
