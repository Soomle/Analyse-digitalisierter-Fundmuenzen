import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.svm import OneClassSVM
import joblib
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QProgressBar, QTextEdit, QFileDialog
from PyQt6.QtCore import QThread, pyqtSignal
import zipfile


class Worker(QThread):
    update_progress = pyqtSignal(int)
    update_text = pyqtSignal(str)

    def __init__(self, zip_file_name, save_path):
        super().__init__()
        self.zip_file_name = zip_file_name
        self.save_path = save_path

    def run(self):
        # Extract ZIP file
        with zipfile.ZipFile(self.zip_file_name, 'r') as zip_ref:
            zip_ref.extractall('images')
            self.update_text.emit('ZIP file extracted.')

        # Initialize the MobileNetV2 model excluding the top layer
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        model = tf.keras.Sequential([base_model, global_average_layer])

        # Process and extract features from images
        features = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.tif']
        for root, _, files in os.walk('images'):
            for file in files:
                if file.lower().endswith(tuple(valid_extensions)):
                    img_path = os.path.join(root, file)
                    image = Image.open(img_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image_resized = image.resize((224, 224))
                    image_array = np.array(image_resized)
                    image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
                    feature = model.predict(np.expand_dims(image_preprocessed, axis=0)).flatten()
                    features.append(feature)
                    self.update_text.emit(f'Processed image: {img_path}')

        # Train a model using OneClassSVM
        classifier = OneClassSVM(gamma='auto')
        classifier.fit(features)
        self.update_text.emit('Model trained.')

        # Save the model to the selected path
        joblib.dump(classifier, self.save_path)
        self.update_text.emit(f'Model saved as {self.save_path}.')


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 300)
        self.setWindowTitle('Image Processing and Model Generation')

        layout = QVBoxLayout()

        self.uploadBtn = QPushButton('Upload ZIP File')
        self.uploadBtn.clicked.connect(self.upload_file)
        layout.addWidget(self.uploadBtn)

        self.textArea = QTextEdit()
        self.textArea.setReadOnly(True)
        layout.addWidget(self.textArea)

        self.setLayout(layout)

        self.zip_file_name = ''
        self.save_path = ''

    def upload_file(self):
        self.zip_file_name, _ = QFileDialog.getOpenFileName(self, "Select ZIP file", "", "Zip Files (*.zip)")
        if self.zip_file_name:
            self.textArea.append(f'File selected: {self.zip_file_name}')
            self.select_save_path()

    def select_save_path(self):
        self.save_path, _ = QFileDialog.getSaveFileName(self, "Select Save Path", "", "Pickle Files (*.pkl)")
        if self.save_path:
            self.textArea.append(f'Model will be saved to: {self.save_path}')
            self.start_processing()

    def start_processing(self):
        if self.zip_file_name and self.save_path:
            self.worker = Worker(self.zip_file_name, self.save_path)
            self.worker.update_text.connect(self.textArea.append)
            self.worker.start()
        else:
            self.textArea.append('Error: No ZIP file selected or save path not specified.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = AppDemo()
    demo.show()
    sys.exit(app.exec())