# import necessary libraries

from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None

    def build_model(self, lr, dropout_rate, fine_tune_at=None):
        # Load ResNet50 with pre-trained ImageNet weights, excluding the top classification layer
        base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the layers of the base model up to the specified fine-tuning layer
        if fine_tune_at is not None:
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True
        else:
            for layer in base_model.layers:
                layer.trainable = False

        # Add custom layers on top of ResNet50
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        output = Dense(3, activation='softmax')(x)

        # Define the complete model
        model = Model(inputs=base_model.input, outputs=output)

        # Compile the model
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()

        self.model = model

    def train_model(self, epochs, batch_size):
        # Add EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )

        return history

    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('accuracy_curve_resnet152.png')
        plt.show()

    def plot_loss(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        #plt.savefig('loss_curve_resnet152.png')
        plt.show()

    def save_model(self, path):
        self.model.save(path)



    
