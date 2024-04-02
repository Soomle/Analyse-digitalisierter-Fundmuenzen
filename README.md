# Analyse digitalisierter mittelalterlicher Fundmünzen

"CoinVision: Coin Image Classifier & Trainer" is a comprehensive software application designed for the analysis and classification of digitalized medieval coin images. The goal of this system is to develop an efficient method for categorizing images based on visual characteristics such as color, shape, size, and texture. CoinVision aims to create a model that can discern these differences and use them to sort images into various categories.

The software is structured into several key components that work seamlessly together:

Data Collection: Users can upload a ZIP file containing a diverse set of images for analysis. The application is not limited to coin images and can handle a wide range of visual materials.

Image Processing: CoinVision includes a preprocessing module that prepares images for feature extraction through operations such as cropping, scaling, converting, and filtering. This step ensures that the images are optimized for the subsequent feature extraction process.

Feature Extraction: The application utilizes scripts and functions to extract relevant features from the images, including shape, color, texture, and edges. These features are then used to train a model that can recognize and classify images based on these visual characteristics.

Model Training and Generation: CoinVision incorporates a model training component that allows users to generate a OneClassSVM model using the extracted features from a series of images. The trained model is then saved to a specified location for future use in classifying new images.

Classification: The core of CoinVision is its classification module, which uses the trained models to categorize uploaded images. Users can load multiple models and compare their classification results, providing a robust system for analyzing and understanding the visual content of the images.

User Interface: CoinVision features a graphical user interface (GUI) that allows users to interact with the application easily. Through the GUI, users can upload images and models, initiate the training process, and view classification results. The interface also includes functionalities for saving the results in HTML format and accessing help and usage instructions.

CoinVision represents a unified solution for historical numismatic analysis, combining advanced computer vision and machine learning techniques to provide a practical tool for researchers and enthusiasts alike.

