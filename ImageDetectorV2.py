import sys
import os
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.svm import OneClassSVM
import joblib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QFileDialog, QLabel, QCheckBox, QSpinBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QThread


class Worker(QThread):
    update_text = pyqtSignal(str)

    def __init__(self, zip_file_name, save_path, use_data_augmentation, augmentation_count=1):
        super().__init__()
        self.zip_file_name = zip_file_name
        self.save_path = save_path
        self.use_data_augmentation = use_data_augmentation
        self.augmentation_count = augmentation_count

    def run(self):
        with zipfile.ZipFile(self.zip_file_name, 'r') as zip_ref:
            zip_ref.extractall('images')
            self.update_text.emit('ZIP file extracted.')

        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        model = tf.keras.Sequential([base_model, global_average_layer])

        if self.use_data_augmentation:
            data_augmentation = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
            self.update_text.emit('Data augmentation enabled.')

        features = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.tif']
        for root, _, files in os.walk('images'):
            for file in files:
                if file.lower().endswith(tuple(valid_extensions)):
                    img_path = os.path.join(root, file)
                    image = Image.open(img_path).convert('RGB')
                    image_resized = image.resize((224, 224))
                    image_array = np.array(image_resized)
                    image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

                    if self.use_data_augmentation:
                        augmented_images = data_augmentation.flow(np.expand_dims(image_array, axis=0), batch_size=1)
                        for _ in range(self.augmentation_count):
                            augmented_image = next(augmented_images)[0]
                            feature = model.predict(np.expand_dims(augmented_image, axis=0)).flatten()
                            features.append(feature)
                    else:
                        feature = model.predict(np.expand_dims(image_preprocessed, axis=0)).flatten()
                        features.append(feature)

                    self.update_text.emit(f'Processed image: {img_path}')

        if features:
            classifier = OneClassSVM(gamma='auto')
            classifier.fit(features)
            self.update_text.emit('Model trained.')

            joblib.dump(classifier, self.save_path)
            self.update_text.emit(f'Model saved as {self.save_path}.')

        shutil.rmtree('images', ignore_errors=True)


class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier with Model Training")
        self.setGeometry(100, 100, 800, 600)
        self.models = []
        self.setup_ui()

    def setup_ui(self):
        # 设置中心组件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 创建并添加"上传ZIP文件"按钮
        self.upload_zip_btn = QPushButton("Upload ZIP for Training")
        self.upload_zip_btn.clicked.connect(self.upload_zip)
        self.layout.addWidget(self.upload_zip_btn)

        # 创建并添加数据增强复选框
        self.data_augmentation_checkbox = QCheckBox("Enable Data Augmentation")
        self.layout.addWidget(self.data_augmentation_checkbox)

        # 创建并添加数据增强次数选择器
        self.augmentation_count_spinbox = QSpinBox()
        self.augmentation_count_spinbox.setMinimum(1)
        self.augmentation_count_spinbox.setMaximum(20)
        self.augmentation_count_spinbox.setValue(1)
        # 数据增强复选框未选中时，增强次数选择器不可用
        self.augmentation_count_spinbox.setEnabled(False)
        self.layout.addWidget(self.augmentation_count_spinbox)

        # 数据增强复选框的状态改变时，启用或禁用增强次数选择器
        self.data_augmentation_checkbox.stateChanged.connect(
            lambda: self.augmentation_count_spinbox.setEnabled(self.data_augmentation_checkbox.isChecked())
        )

        # 创建并添加模型数量选择器
        self.model_count_spinbox = QSpinBox()
        self.model_count_spinbox.setMinimum(1)  # 最小模型数量
        self.model_count_spinbox.setMaximum(5)  # 最大模型数量
        self.model_count_spinbox.setValue(1)    # 初始模型数量
        self.layout.addWidget(self.model_count_spinbox)
        self.model_count_spinbox.valueChanged.connect(self.set_model_count)

        # 创建并添加"上传图片进行分类"按钮
        self.upload_image_btn = QPushButton("Upload Image for Classification")
        self.upload_image_btn.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_image_btn)

        # 创建并添加结果显示区域
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.layout.addWidget(self.result_area)

        # 创建并添加图片展示标签
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        # 根据初始模型数量设置创建模型上传按钮
        self.create_model_upload_buttons(self.model_count_spinbox.value())
    def upload_zip(self):
        zip_file_name, _ = QFileDialog.getOpenFileName(self, "Select ZIP file for Training", "", "Zip Files (*.zip)")
        if zip_file_name:
            self.result_area.append(f'ZIP file selected: {zip_file_name}')
            self.select_save_path_for_training(zip_file_name)

    def select_save_path_for_training(self, zip_file_name):
        save_path, _ = QFileDialog.getSaveFileName(self, "Select Save Path for Model", "", "Model Files (*.pkl)")
        if save_path:
            self.result_area.append(f'Model will be saved to: {save_path}')
            self.start_training(zip_file_name, save_path)

    def start_training(self, zip_file_name, save_path):
        use_data_augmentation = self.data_augmentation_checkbox.isChecked()
        augmentation_count = self.augmentation_count_spinbox.value() if use_data_augmentation else 1
        self.worker = Worker(zip_file_name, save_path, use_data_augmentation, augmentation_count)
        self.worker.update_text.connect(self.result_area.append)
        self.worker.start()

    def set_model_count(self, value):
        self.models = [None] * value  # Adjust the model list according to the spinbox value
        # Regenerate model upload buttons based on the new model count
        self.create_model_upload_buttons(value)

    def create_model_upload_buttons(self, model_count):
        # First, remove old buttons if any
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if isinstance(widget, QPushButton) and 'Upload Model' in widget.text():
                self.layout.removeWidget(widget)
                widget.deleteLater()

        # Then, add new upload buttons for each model
        for i in range(1, model_count + 1):
            btn = QPushButton(f"Upload Model {i}")
            btn.clicked.connect(lambda _, b=i - 1: self.upload_model(b))
            self.layout.insertWidget(self.layout.count() - 3, btn)  # Insert above the image upload button

    def upload_model(self, model_index):
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
            self.show_image(image_filename)
            self.process_image(image_filename)

    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

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

    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def predict_image(self, model, image):
        image_array = np.array(image)
        image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        feature = self.feature_extractor.predict(np.expand_dims(image_preprocessed, axis=0)).flatten()
        score = model.decision_function([feature])[0]
        return score

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier with Model Training")
        self.setGeometry(100, 100, 800, 600)
        self.models = []
        self.setup_ui()
        self.feature_extractor = self.initialize_feature_extractor()  # 初始化特征提取器

    def calculate_percentages(self, scores):
        exp_scores = [np.exp(score) for score in scores]
        sum_exp_scores = sum(exp_scores)
        percentages = [exp_score / sum_exp_scores for exp_score in exp_scores]
        return percentages

    def display_percentages(self, percentages):
        self.result_area.clear()  # Clear previous results
        for i, percentage in enumerate(percentages, start=1):
            self.result_area.append(f"Model {i}: {percentage:.2%}")

    @staticmethod
    def initialize_feature_extractor():
        feature_extractor = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')
        return tf.keras.Sequential([feature_extractor, tf.keras.layers.GlobalAveragePooling2D()])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = ImageClassifierApp()
    mainWin.show()
    sys.exit(app.exec())