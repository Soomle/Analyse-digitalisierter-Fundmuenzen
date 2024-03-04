import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.svm import OneClassSVM
import joblib
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QFileDialog, QLabel

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier with Multiple Models")
        self.setGeometry(100, 100, 400, 300)
        self.models = []
        self.setup_ui()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.upload_model_btns = []
        for i in range(1, 4):
            btn = QPushButton(f"Upload Model {i}")
            btn.clicked.connect(lambda _, b=i-1: self.upload_model(b))
            self.layout.addWidget(btn)
            self.upload_model_btns.append(btn)

        self.upload_image_btn = QPushButton("Upload Image")
        self.upload_image_btn.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_image_btn)

        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.layout.addWidget(self.result_area)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.feature_extractor = self.initialize_feature_extractor()

    @staticmethod
    def initialize_feature_extractor():
        feature_extractor = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')
        return tf.keras.Sequential([feature_extractor, tf.keras.layers.GlobalAveragePooling2D()])

    def upload_model(self, model_index):
        if len(self.models) <= model_index:
            self.models.append(None)
        model_filename, _ = QFileDialog.getOpenFileName(self, f"Select Model {model_index + 1} File", "", "Model Files (*.pkl)")
        if model_filename:
            try:
                self.models[model_index] = joblib.load(model_filename)
                self.result_area.append(f"Model {model_index + 1} loaded successfully.")
            except Exception as e:
                self.result_area.append(f"Failed to load Model {model_index + 1}: {e}")

    def upload_image(self):
        image_filename, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Image Files (*.jpg *.jpeg *.png *.tif)")
        if image_filename:
            self.process_image(image_filename)

    def process_image(self, image_filename):
        if not all(self.models):
            self.result_area.append("Please upload all models first.")
            return

        try:
            image = Image.open(image_filename).convert('RGB').resize((224, 224))
            scores = [self.predict_image(model, image) for model in self.models]
            percentages = self.calculate_percentages(scores)
            self.display_percentages(percentages)
        except Exception as e:
            self.result_area.append(f"Failed to process image: {e}")

    def predict_image(self, model, image):
        image_array = np.array(image)
        image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        feature = self.feature_extractor.predict(np.expand_dims(image_preprocessed, axis=0)).flatten()
        score = model.decision_function([feature])[0]
        return score

    def calculate_percentages(self, scores):
        exp_scores = [np.exp(score) for score in scores]
        sum_exp_scores = sum(exp_scores)
        percentages = [exp_score / sum_exp_scores for exp_score in exp_scores]
        return percentages

    def display_percentages(self, percentages):
        self.result_area.clear()  # Clear previous results
        for i, percentage in enumerate(percentages, start=1):
            self.result_area.append(f"Model {i}: {percentage:.2%}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = ImageClassifierApp()
    mainWin.show()
    sys.exit(app.exe`xxxxxxxc())