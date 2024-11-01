# import necessary libraries

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

class Trainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None

    def build_model(self, lr, dropout_rate, trainable_layers):
        # Load the VGG16 model, excluding the top layers (i.e., without the final dense layers)
        vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

         # Freeze all layers initially
        for layer in vgg.layers:
          layer.trainable = False

        # Unfreeze the last 'trainable_layers' layers of the VGG16 model
        for layer in vgg.layers[-trainable_layers:]:
          layer.trainable = True

        # Add custom layers on top of the VGG16 base
        x = Flatten()(vgg.output)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)  # Add Batch Normalization after Dense layer
        x = Dropout(dropout_rate)(x)  # Add Dropout to prevent overfitting
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)  # Add Batch Normalization
        x = Dropout(dropout_rate)(x)  # Add another Dropout layer
        output_layer = Dense(3, activation='softmax')(x)  # Assuming 3 classes for classification

        # Compile the model
        model = Model(inputs=vgg.input, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        self.model = model

    def train_model(self, epochs, batch_size):
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.3,
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
        plt.savefig('accuracy_without_tuning_vgg16_100epochs.png')
        plt.show()

    def plot_loss(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('loss_without_tuning_vgg16_100epochs.png')
        plt.show()

    def save_model(self, path):
        self.model.save(path)

