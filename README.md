**Face Mask Recognition Using CNN, Inception, VGG, and ResNet**

This repository contains the implementation of face mask recognition using four deep learning architectures: CNN, Inception, VGG, and ResNet.
The models are trained to classify faces into three categories: Correct Mask, Incorrect Mask, and No Mask. 

**Project Overview**
This project implements face mask recognition by leveraging several state-of-the-art deep learning architectures. The models are designed to classify images into the following categories:

Correct Mask: When the mask is worn properly.
Incorrect Mask: When the mask is worn incorrectly.
No Mask: When no mask is worn.
By using transfer learning from pre-trained models (Inception, VGG, and ResNet), we fine-tuned the networks for mask detection tasks, improving the speed and accuracy of classification.

**Models Used**
Custom CNN: A simple Convolutional Neural Network built from scratch for baseline performance.
Inception: Google's Inception model (InceptionV3), known for its efficient layer architecture and strong feature extraction capabilities.
VGG: VGG16, a deep network with uniform layers, providing strong performance on image classification tasks.
ResNet: Residual Network (ResNet50), using skip connections to build deeper networks without vanishing gradients.

**Dataset**
The dataset consists of labeled images in three categories: Correct Mask, Incorrect Mask, and No Mask.
Each model was trained using transfer learning techniques with image augmentation to improve robustness.
