#import necessary libraries

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

class Tester:

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.model = None  # Initialize model as None

    def load_model(self, path):
        self.model = load_model(path)
        print(f"Model loaded from {path}.")
        return self.model

    # Evaluate the model on the test dataset
    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before evaluation.")
        score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def evaluate_through_metrics(self):
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before evaluation.")

        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)

        print("Accuracy:", accuracy_score(y_true_classes, y_pred_classes))
        print("Classification Report:\n", classification_report(y_true_classes, y_pred_classes))
        print("Confusion Matrix:\n", confusion_matrix(y_true_classes, y_pred_classes))

    # Predict the results for the first 5 images in the test dataset.
    def predict_on_test_data(self):
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before making predictions.")

        # Predict the labels for the first 5 test images
        y_pred = self.model.predict(self.X_test[:5])
        predicted_classes = np.argmax(y_pred, axis=1)
        actual_classes = np.argmax(self.y_test[:5], axis=1)
        confidence_scores = np.max(y_pred, axis=1)

        # Class names
        class_names = ["Correct Mask", "Incorrect Mask", "No Mask"]

        # Plot the first 5 X_test images along with actual and predicted labels
        plt.figure(figsize=(15, 15))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.imshow(self.X_test[i])
            plt.title(f"Actual: {class_names[actual_classes[i]]}\n"
                      f"Predicted: {class_names[predicted_classes[i]]}\n"
                      f"Confidence: {confidence_scores[i]:.2f}")

            plt.axis('off')
        plt.tight_layout()
        #plt.savefig('predictions_resnet152.png')
        plt.show()

