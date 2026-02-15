# Skin-Cancer-Detection-Hybrid-Model
## A Deep Learning approach to Skin Cancer Detection using a Hybrid Parallel Model for Melanoma Classification.

Berserker: A Multi-Stream Hybrid Architecture for Advanced Melanoma Diagnosis
Berserker is a cutting-edge deep learning framework that redefines skin cancer classification by integrating Convolutional Neural Networks (CNNs) and Hierarchical Vision Transformers (Swin Transformer) into a unified parallel pipeline.
Unlike traditional single-backbone models, this architecture employs a Multi-Stream Feature Fusion strategy. It simultaneously extracts local morphological textures through CNNs and global contextual dependencies via the Swin Transformer's shifted-window mechanism. By leveraging this hybrid approach, Berserker ensures a robust representation of malignant features, capturing both fine-grained skin lesion patterns and complex spatial relationships that are often missed by standard models.

## Technical Implementation:
Language: Python 3.8+
Deep Learning Framework: PyTorch (chosen for its dynamic computational graph and research flexibility).
Pre-trained Backbones: torchvision and timm (PyTorch Image Models).
Augmentation Pipeline: Albumentations (used for high-performance spatial and pixel-level transformations to handle class imbalance).
Experiment Tracking: Matplotlib & Seaborn for visualizing loss/accuracy curves and confusion matrices.

## The model is implemented as a parallel ensemble class. Each input image is concurrently processed by:
ResNet50 & DenseNet201: Traditional CNNs that focus on hierarchical texture and edge detection.
EfficientNetB4: A compound-scaled CNN that balances depth and width.
Swin-B Transformer: A hierarchical Vision Transformer that captures long-range dependencies using Shifted Windows.
The 1D feature vectors from these four streams are concatenated into a Unified Feature Representation, followed by a Dropout layer (to prevent overfitting) and a Softmax output layer for binary classification (Melanoma vs. Benign).

## Dataset Information:
This project utilizes datasets provided by the International Skin Imaging Collaboration (ISIC), specifically focusing on the ISIC 2019 and ISIC 2020 challenges.

## Dataset Composition
Training Set: A combined and balanced distribution of images from both years (2019-2020).
Preprocessing: All images were normalized and resized to 224x224 pixels.
Class Balance: To address the inherent imbalance in medical data, we applied strategic Oversampling and Data Augmentation techniques (Horizontal Flip, Vertical Flip, Random Rotation, and Color Jitter).

## ISIC License Compliance:
### To strictly follow the ISIC Terms of Use and respect data privacy/licensing:
No Data Redistribution: This repository does not host the raw image files.
How to Access: Users must register and download the data directly from the ISIC Archive.
Use Case: The code provided is for research and educational purposes only.
Attribution: All credit for the original dermoscopy images belongs to the contributors of the ISIC challenges.
