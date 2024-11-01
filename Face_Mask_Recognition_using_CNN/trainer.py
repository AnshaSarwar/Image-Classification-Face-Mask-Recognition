# import necessary libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from dataset import Dataset
import matplotlib.pyplot as plt
plt.ion()

# class for training the model
class Trainer:

    def __init__(self, X_train, y_train, X_test, y_test):

        # Initialize Trainer with train and test data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None  # Initialize model attribute to None

    def build_model(self, lr, dropout2_rate, num_filters, kernel_size, batch_size, n_epochs):
        # Define the model architecture
        model = Sequential()
        model.add(Conv2D(num_filters, (kernel_size, kernel_size), activation='relu', input_shape=(224, 224, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(num_filters * 2, (kernel_size, kernel_size), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(num_filters * 4, (kernel_size, kernel_size), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout2_rate))
        model.add(BatchNormalization())
        model.add(Dense(3, activation='softmax'))

        # Compile the model
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()

        self.model = model

    def train_model(self, epochs, batch_size):
        # Add EarlyStopping callback to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(self.X_train, self.y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

        return history

    # plot training and validation accuracy
    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        #plt.savefig('accuracy.png')
        plt.show(block=True)

    # plot training and validation loss
    def plot_loss(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        #plt.savefig('loss.png')
        plt.show(block=True)

    # Save the trained model to a file
    def save_model(self, path):
        self.model.save(path)

