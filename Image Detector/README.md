Image Classification with One-Class SVM and MobileNetV2 Feature Extraction
This project uses a machine learning approach to classify images, determining whether they belong to a specific class as defined by a One-Class SVM (Support Vector Machine) model. It leverages the MobileNetV2 model from TensorFlow's Keras applications for feature extraction.
Prerequisites
Before running this script, ensure you have the following installed:
Python 3.6 or later
TensorFlow 2.x
NumPy
Scikit-Learn
Pillow (PIL Fork)
Joblib
If you are using Google Colab, these packages should already be available.
Installation
To install the required Python packages, you can use pip:
shCopy code
pip install numpy tensorflow scikit-learn pillow joblib
Usage
Prepare your One-Class SVM model: Train a One-Class SVM model on your dataset and save it using joblib.dump(). The model should be trained to distinguish images from a single class of interest from all other images.
Upload your model and images:
When prompted, upload your trained One-Class SVM model file (e.g., model.joblib).
Then, upload a ZIP file containing the images you want to classify.
Run the script: Execute the script in your Python environment. If using Google Colab, simply run the cells in order.
View results: The script will print out each image's file name, whether it belongs to the model's class, and the decision function score.
Customization
You can adjust the custom_threshold variable to change how strict the classifier is. A lower threshold value means the model is less likely to reject images.
Features
Model Upload: Upload your own trained One-Class SVM model.
Image Processing: Automatically resizes and preprocesses images for MobileNetV2 compatibility.
Feature Extraction: Uses MobileNetV2 for robust feature extraction.
Custom Threshold: Allows for adjusting the sensitivity of the classifier.
Batch Prediction: Classifies a batch of images by processing a ZIP file upload.