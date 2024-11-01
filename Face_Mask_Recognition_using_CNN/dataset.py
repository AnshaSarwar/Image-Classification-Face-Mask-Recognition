# import necessary libraries
import numpy as np
import os
import matplotlib.pyplot as plt 
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# Define the class for the dataset
class Dataset:
  def __init__(self):

    # Define the path for the dataset
    self.dataset_path = "dataset"

    # List the contents of each sub-folder (correct, incorrect, nomask)
    self.correct_path = "dataset/Correct"
    self.incorrect_path = "dataset/Incorrect"
    self.nomask_path = "dataset/NoMask"
    

  # Load images
  def load_images(self, path, label):
    images = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            img = image.load_img(os.path.join(path, filename), target_size=(224, 224, 3))
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return images, labels


  # Display images
  def show_images(self, images, labels, class_label, num_images=5):

    # Get indices of images with the desired class label
    indices = []
    for i, label in enumerate(labels):
        if np.argmax(label) == class_label:
            indices.append(i)

    # Display up to 'num_images' images for the desired class label
    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(indices))):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[indices[i]])
        plt.title(f"Label: {class_label}")
        plt.axis('off')
    plt.show()


  # Load and preprocess the dataset
  def load_and_preprocess_data(self):

    # Load the images and labels
    X, y = [], []

    for path, label in [(self.correct_path, 0), (self.incorrect_path, 1), (self.nomask_path, 2)]:
            images, labels = self.load_images(path, label)
            X.extend(images)
            y.extend(labels)

    # Preprocess the data
    # Normalize the pixel values
    X = np.array(X) / 255

    # One-hot encode the labels
    y = to_categorical(y, num_classes=3)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test
