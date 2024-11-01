# import necessary libraries
from bayes_opt import BayesianOptimization
from hyperopt import fmin, tpe, hp, Trials
from dataset import Dataset
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

# Class for hyperparameter Tuning
class Optimizer:

# Initialize the Optimizer with training and testing data
  def __init__(self, X_train, y_train, X_test, y_test):
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test


  def optimize(self):

      # Define the hyperparameter search space
      space = {
            'lr': hp.choice('lr', [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.1, 1.0]),
            'dropout2_rate': hp.choice('dropout2_rate', [0.1, 0.2, 0.3, 0.4, 0.5]),
            'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
            'num_filters': hp.choice('num_filters', [16, 32, 64, 128]),
            'kernel_size': hp.choice('kernel_size', [3, 4, 5, 6, 7]),
            'n_epochs': hp.choice('n_epochs', [10, 20, 30])
      }

      # Function to train and evaluate the model with given hyperparameters
      def train_and_evaluate(params):
          model = Sequential()
          model.add(Conv2D(int(params['num_filters']), (int(params['kernel_size']), int(params['kernel_size'])), activation='relu', input_shape=(224, 224, 3)))
          model.add(BatchNormalization())
          model.add(MaxPooling2D((2, 2)))
          model.add(Conv2D(int(params['num_filters']) * 2, (int(params['kernel_size']), int(params['kernel_size'])), activation='relu'))
          model.add(BatchNormalization())
          model.add(MaxPooling2D((2, 2)))
          model.add(Conv2D(int(params['num_filters']) * 4, (int(params['kernel_size']), int(params['kernel_size'])), activation='relu'))
          model.add(BatchNormalization())
          model.add(MaxPooling2D((2, 2)))
          model.add(Flatten())
          model.add(Dense(128, activation='relu'))
          model.add(Dropout(params['dropout2_rate']))
          model.add(BatchNormalization())
          model.add(Dense(3, activation='softmax'))

          # Compile and train the model
          optimizer = Adam(learning_rate=params['lr'])
          model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
          history = model.fit(self.X_train, self.y_train, epochs=int(params['n_epochs']), batch_size=int(params['batch_size']), validation_split=0.2, verbose=2)
          val_accuracy = max(history.history['val_accuracy'])

          # Return the negative validation accuracy (for minimization)
          return -val_accuracy

      # Set up the Trials object for tracking the optimization process
      trials = Trials()

      # Run the hyperparameter optimization
      best = fmin(fn=train_and_evaluate, space=space, algo=tpe.suggest, max_evals=1, trials=trials)
      return best

