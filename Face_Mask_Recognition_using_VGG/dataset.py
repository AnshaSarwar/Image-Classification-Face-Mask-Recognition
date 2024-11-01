# import necessary libraries
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Dataset:
    def __init__(self):
        # Define the base path for the extracted dataset
        self.dataset_path = "dataset"

        # List the contents of each sub-folder (correct, incorrect, nomask)
        self.correct_path = "dataset/Correct"
        self.incorrect_path = "dataset/Incorrect"
        self.nomask_path = "dataset/NoMask"

        # Create an instance of ImageDataGenerator with geometric augmentations
        self.datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.3,
            zoom_range=[0.8, 1.2],
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            rescale=1.0/255
        )

        # Path for caching augmented data
        self.cache_path = 'class_2_augmented_data_vgg16.pkl'

    def load_images(self, path, label):
        images = []
        labels = []
        for filename in os.listdir(path):
            if filename.endswith('.jpg'):
                img = image.load_img(os.path.join(path, filename), target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
                images.append(img_array)
                labels.append(label)
        return images, labels

    # Augment class 2 data once and cache it
    def augment_class_2(self, augment_factor):
        if not os.path.exists(self.cache_path):  # Only augment if cache doesn't exist
            print("Augmenting Class 2 data...")

            # Load original class 2 data
            X_class_2, y_class_2 = self.load_images(self.nomask_path, 2)
            augmented_images = []
            augmented_labels = []

            # Augment each image in Class 2 'augment_factor' times
            for img in X_class_2:
                for _ in range(augment_factor):
                    augmented_img = self.datagen.random_transform(img)
                    augmented_images.append(augmented_img)
                    augmented_labels.append(2)  # Label for class 2

            # Save the augmented data to the cache
            with open(self.cache_path, 'wb') as f:
                pickle.dump((augmented_images, augmented_labels), f)

            print(f"Class 2 data augmented and cached at {self.cache_path}")

        else:
            print(f"Augmented Class 2 data already exists in cache at {self.cache_path}")

    # Load the cached augmented class 2 data
    def load_cached_class_2_data(self):
        with open(self.cache_path, 'rb') as f:
            augmented_images, augmented_labels = pickle.load(f)
        return augmented_images, augmented_labels

    def load_and_preprocess_data(self, augment_class_2=False, augment_factor=2):
        X, y = [], []

        # Load and append images for Class 0 and Class 1 (no augmentation)
        for path, label in [(self.correct_path, 0), (self.incorrect_path, 1)]:
            images, labels = self.load_images(path, label)
            X.extend(images)
            y.extend(labels)

        # Check if cache exists
        if not os.path.exists(self.cache_path):
            if augment_class_2:
                print("Cache not found. Augmenting Class 2...")
                self.augment_class_2(augment_factor)
            else:
                raise FileNotFoundError("No cached data for Class 2. Please run with augment_class_2=True first.")
        else:
            print("Using cached Class 2 data.")

        # Load augmented data for Class 2 from cache
        X_class_2, y_class_2 = self.load_cached_class_2_data()

        # Append Class 2 data
        X.extend(X_class_2)
        y.extend(y_class_2)

        # Convert lists to arrays
        X = np.array(X)
        y = np.array(y)

        # One-hot encode the labels
        y = to_categorical(y, num_classes=3)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

        return X_train, X_test, y_train, y_test

    def get_class_distribution(self, labels):
        # Convert one-hot encoded labels to class indices
        class_indices = np.argmax(labels, axis=1)

        # Get unique classes and their counts
        unique, counts = np.unique(class_indices, return_counts=True)

        # Create a dictionary to map class indices to counts
        distribution = dict(zip(unique, counts))

        return distribution


