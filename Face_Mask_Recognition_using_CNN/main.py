from dataset import Dataset
from trainer import Trainer
from tester import Tester
from optimizer import Optimizer

# import necessary libraries
import numpy as np
import os
import matplotlib.pyplot as plt 
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from bayes_opt import BayesianOptimization
from hyperopt import fmin, tpe, hp, Trials
plt.ion()


if __name__ == '__main__':

  # Load and preprocess the dataset
  dataset = Dataset()
  X_train, X_test, y_train, y_test = dataset.load_and_preprocess_data()

  # Print the shapes of the train and test sets
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

  # # Show images for each class in the training set (0 = Correct, 1 = Incorrect, 2 = NoMask)
  # print("Showing 'Correct' mask images:")
  # dataset.show_images(X_train, y_train, class_label=0)

  # print("Showing 'Incorrect' mask images:")
  # dataset.show_images(X_train, y_train, class_label=1)

  # print("Showing 'NoMask' images:")
  # dataset.show_images(X_train, y_train, class_label=2)


  # Initialize Trainer and build the model
  trainer = Trainer(X_train, y_train, X_test, y_test)
  trainer.build_model(lr=0.000001, dropout2_rate=0.1, num_filters=16, kernel_size=4, batch_size=16, n_epochs=1)

  # Train the model
  history = trainer.train_model(epochs=1, batch_size=16)

  # # Plot training and validation accuracy
  # trainer.plot_accuracy(history)

  # # Plot training and validation loss
  # trainer.plot_loss(history)

  '''Model is saved already, can use this function call for your own tasks'''
  # Save the trained model     
  #trainer.save_model('model.h5')

  # Ensure you have a model loaded or trained
  # Replace 'model_path' with the actual path to your saved model
  model_path = 'model.h5'
  tester = Tester(X_test=X_test, y_test=y_test)
  tester.load_model(model_path)
    
  tester.evaluate_model()
  tester.evaluate_through_metrics()
  tester.predict_on_test_data()

  '''Can use for your own task'''
  # hyperparameter tuning
  # optimizer = Optimizer(X_train, y_train, X_test, y_test)
  # best_params = optimizer.optimize()
  # print(f"Best parameters found: {best_params}")

