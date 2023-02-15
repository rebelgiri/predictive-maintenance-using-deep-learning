from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from  timeseries_classfication import diVibes
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import argparse
import os
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # Basic parameters
    parser.add_argument('--model_name', type=str, default='FCN', help='The name of the model.')
    parser.add_argument('--data_dir', type=str, default='data', help='The path of the training and testing dataset.')
    parser.add_argument('--output_dir', type=str, default='output', help='The path of the output directory.')
    parser.add_argument('--N', type=int, default=100000, help='The length of timeseries slice.')
    parser.add_argument('--use_saved_data', type=int, default=0, help='Load the saved training and test dataset.')
    

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    
    output_dir = os.path.join(args.output_dir, datetime.strftime(datetime.now(), '%m%d-%H%M%S'))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_dir = "logs/fit/"
    log_dir = os.path.join(output_dir, log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if not args.use_saved_data:
        trainer = diVibes()
        training_dataset_path = os.path.join(args.data_dir, 'training')
        class_names = trainer.get_dataset_info(training_dataset_path)
        print('Classes in training dataset {0}'.format(class_names))
    else:
        with open(os.path.join(args.data_dir, 'class_name.txt'), 'r') as f:
            class_names = f.readline().split()
        trainer = diVibes(class_names=class_names)    

    # Load training dataset
    if not args.use_saved_data:
        training_dataset, training_labels = trainer.get_dataset(training_dataset_path)
    else:
        training_dataset = np.load(os.path.join(args.data_dir, 'training_dataset.npy'))
        training_labels = np.load(os.path.join(args.data_dir, 'training_labels.npy'))

    # Load test dataset
    if not args.use_saved_data:
        test_dataset_path = os.path.join(args.data_dir, 'test')
        test_dataset, test_labels = trainer.get_dataset(test_dataset_path)
    else:
        test_dataset = np.load(os.path.join(args.data_dir, 'test_dataset.npy'))
        test_labels = np.load(os.path.join(args.data_dir, 'test_labels.npy'))

    print('Size of training dataset {}'.format(training_dataset.shape))
    training_labels = np.array(training_labels)
    print('Size of training dataset labels {}'.format(training_labels.shape))

    print('Size of test dataset {}'.format(test_dataset.shape))
    test_labels = np.array(test_labels)
    print('Size of test dataset labels {}'.format(test_labels.shape))
    
    # Visualize the data. Here we visualize one timeseries example for each class in the dataset.
    classes = np.unique(training_labels, axis=0)
  
    plt.figure()
    for c in classes:
        c_x_train = training_dataset[training_labels == c]
        c_x_train = c_x_train.reshape(training_dataset[training_labels == c].shape[0], 3, training_dataset.shape[1])
        plt.plot(c_x_train[0][0][0:500], label="class " + str(c))
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, '500_x-direction_samples.png'))
    plt.close() 


    plt.figure()
    for c in classes:
        c_x_train = training_dataset[training_labels == c]
        c_x_train = c_x_train.reshape(training_dataset[training_labels == c].shape[0], 3, training_dataset.shape[1])
        plt.plot(c_x_train[0][0][0:100], label="class " + str(c))
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, '100_x-direction_samples.png'))
    plt.close() 


    plt.figure()
    for c in classes:
        c_x_train = training_dataset[training_labels == c]
        c_x_train = c_x_train.reshape(training_dataset[training_labels == c].shape[0], 3, training_dataset.shape[1])
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
    model_name = args.model_name + '_' + datetime.strftime(datetime.now(), '%m.%d-%H.%M.%S') + '.h5'
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
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
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

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
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
    plt.savefig(os.path.join(output_dir, 'training_and_validation_accuracy.png'))
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




