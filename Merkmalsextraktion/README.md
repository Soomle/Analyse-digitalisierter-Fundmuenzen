Project Name: Image Feature Extraction and Model Training

Introduction

This project is designed to automate the process of feature extraction from images stored in a ZIP file and to train a machine learning model using these features. It utilizes TensorFlow's MobileNetV2 for feature extraction and scikit-learn's OneClassSVM for anomaly detection or novelty detection. The project is implemented in Python and uses PyQt6 for a basic graphical user interface (GUI) that allows users to upload a ZIP file and specify where to save the trained model.

Requirements

To run this project, you need the following dependencies:

- Python 3.x
- NumPy
- Pillow (PIL Fork)
- TensorFlow
- scikit-learn
- Joblib
- PyQt6

These dependencies ensure that the project can process images, extract features, train the model, and provide a user-friendly interface.

Installation

First, make sure you have Python 3.x installed on your system. Then, install the required dependencies using pip:

```bash
pip install numpy Pillow tensorflow scikit-learn joblib PyQt6
```

Usage

To use the application, follow these steps:

1. Launch the program by running the Python script:

    ```bash
    python <script_name>.py
    ```

2. In the GUI, click the "Upload ZIP File" button to select a ZIP file containing the images you want to process.

3. After selecting the ZIP file, choose where to save the trained model by clicking the prompt that appears.

The application will then extract images from the ZIP file, preprocess them, extract features using MobileNetV2, train a OneClassSVM model with these features, and save the model to the specified location.

Code Overview

Worker Class

The `Worker` class is a QThread that handles the heavy lifting of processing images and training the model. It emits signals to update the GUI with progress and messages.

Key Methods:

- `run()`: Main method executed by the thread. It extracts images from the ZIP file, preprocesses them, extracts features, trains the model, and saves it.

AppDemo Class

The `AppDemo` class creates the GUI using PyQt6. It allows users to upload a ZIP file and specify the save location for the trained model.

Key Components:

- `uploadBtn`: Button for uploading the ZIP file.
- `textArea`: Text area for displaying status messages and progress.
- `upload_file()`: Method to handle the ZIP file upload.
- `select_save_path()`: Method to specify the model's save path.
- `start_processing()`: Method to start the `Worker` thread for processing and training.

Main Execution

The script checks if it's the main program and launches the PyQt application, creating an instance of `AppDemo`.
