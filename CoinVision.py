import sys
import os
import cv2
import io
import shutil
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.svm import OneClassSVM
import joblib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QApplication, QMainWindow, QProgressBar, QPushButton, QHBoxLayout, QVBoxLayout, QCheckBox,
                             QWidget, QFileDialog, QTextEdit, QSpinBox, QMenu, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal



def initialize_feature_extractor():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return tf.keras.Sequential([base_model, tf.keras.layers.GlobalAveragePooling2D()])

def tiff_to_base64_png(tiff_image_path):
    with Image.open(tiff_image_path) as img:
        with io.BytesIO() as buffer:
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

class Worker(QThread):
    update_text = pyqtSignal(str)
    update_progress = pyqtSignal(int)

    def __init__(self, zip_file_name, save_path, use_data_augmentation, augmentation_count=1):
        super().__init__()
        self.zip_file_name = zip_file_name
        self.save_path = save_path
        self.use_data_augmentation = use_data_augmentation
        self.augmentation_count = augmentation_count
        self.model = initialize_feature_extractor()
        self.total_images = 0  # 添加属性以存储图片总数

        # 配置数据增强
        if self.use_data_augmentation:
            self.data_gen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

    def run(self):
        try:
            self.update_progress.emit(0)
            self.extract_and_process_images()
            train_features, test_features = self.extract_features('images')
            self.train_and_save_model(train_features, test_features)
        except Exception as e:
            self.update_text.emit(f'Error occurred: {str(e)}')
        finally:
            shutil.rmtree('images', ignore_errors=True)

    def extract_and_process_images(self):
        with zipfile.ZipFile(self.zip_file_name, 'r') as zip_ref:
            zip_ref.extractall('images')
        self.update_text.emit('ZIP file extracted.')
        self.total_images = sum([len(files) for r, d, files in os.walk('images')])  # 计算图片总数

        # 新增代码：区分处理不同类型的图像
        for img_path in self.find_images('images'):
            if "_normal" in img_path:
                self.process_normal_image(img_path)
            elif "_albedo" in img_path:
                self.process_albedo_image(img_path)

    def process_normal_image(self, image):
        # 对于法线图像，通常我们希望增强其细节
        # 可以应用一个高通滤波器来增强边缘
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        high_pass = gray - blurred
        detail_enhanced = cv2.addWeighted(gray, 1.5, high_pass, 0.5, 0)
        return image

    def process_albedo_image(self, image):
        # 对于颜色图像，可能需要调整亮度和对比度来改善视觉效果
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        corrected_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return image

    def extract_and_process_images(self):
        with zipfile.ZipFile(self.zip_file_name, 'r') as zip_ref:
            zip_ref.extractall('images')
        self.update_text.emit('ZIP file extracted.')
        self.total_images = sum([len(files) for r, d, files in os.walk('images')])  # 计算图片总数

    def extract_features(self, image_dir):
        features = []
        processed_count = 0  # 已处理图片数
        for img_path in self.find_images(image_dir):
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image at path: {img_path}")

            # 根据文件名处理不同类型的图像
            if "_normal" in img_path:
                image = self.process_normal_image(image)
            elif "_albedo" in img_path:
                image = self.process_albedo_image(image)

            # 应用数据增强
            if self.use_data_augmentation:
                image = np.expand_dims(image, axis=0)
                augmented_images = self.data_gen.flow(image, batch_size=1)
                for i in range(self.augmentation_count):
                    aug_image = next(augmented_images)[0]
                    feature = self.extract_feature(aug_image)
                    features.append(feature)
            else:
                # 如果未使用数据增强，则直接提取特征
                feature = self.extract_feature(image)
                features.append(feature)

            processed_count += 1
            self.update_progress.emit(int((processed_count / self.total_images) * 100))  # 根据处理的图片数更新进度条

        features_np = np.array(features)
        train_features, test_features = self.split_data(features_np)
        return train_features, test_features

    def find_images(self, directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    yield os.path.join(root, file)

    def preprocess_image(self, image):
        # 调整图像大小到期望的输入尺寸
        image_resized = cv2.resize(image, (224, 224))

        # 应用双边滤波进行去噪，保留边缘信息
        denoised_image = cv2.bilateralFilter(image_resized, d=9, sigmaColor=75, sigmaSpace=75)

        # 归一化图像像素值到 0-1 范围
        normalized_image = denoised_image / 255.0

        # 如果启用数据增强，应用数据增强技术
        if self.use_data_augmentation:
            processed_images = self.apply_data_augmentation(normalized_image)
        else:
            # 如果未启用数据增强，返回归一化后的图像
            processed_images = [normalized_image]

        return processed_images

    def train_and_save_model(self, train_features, test_features):
        # 检查是否有足够的特征来训练模型
        if train_features.size == 0:
            self.update_text.emit('No features to train the model.')
            return

        try:
            # 使用OneClassSVM训练模型
            classifier = OneClassSVM(gamma='scale', nu=0.2)
            classifier.fit(train_features)

            # 保存训练好的模型
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)  # 确保保存路径的目录存在
            joblib.dump(classifier, self.save_path)
            self.update_text.emit(f'Model trained and saved as {self.save_path}.')

            # 使用测试特征评估模型
            test_predictions = classifier.predict(test_features)
            anomaly_ratio = np.mean(test_predictions == -1)
            self.update_text.emit(f'Anomaly ratio in test data: {anomaly_ratio:.2f}')
        except Exception as e:
            self.update_text.emit(f'Error occurred during model training and saving: {str(e)}')

    def apply_data_augmentation(self, image):
        if not self.use_data_augmentation:
            return [image]

        augmented_images = self.data_gen.flow(np.expand_dims(image, 0), batch_size=1)
        return [next(augmented_images)[0] for _ in range(self.augmentation_count)]

    def extract_feature(self, image):
        image_resized = cv2.resize(image, (224, 224))
        image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_resized)
        return self.model.predict(np.expand_dims(image_preprocessed, axis=0)).flatten()

    def split_data(self, data, test_ratio=0.2):
        """
        分割数据为训练集和测试集。
        """
        np.random.shuffle(data)  # 打乱数据
        split_idx = int(len(data) * (1 - test_ratio))
        return data[:split_idx], data[split_idx:]


class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CoinVision: Coin Image Classifier & Trainer")
        self.setGeometry(100, 100, 800, 600)
        self.models = []
        self.model_names = []  # 新增属性以存储模型名称
        self.setup_ui()
        self.setup_menu()
        self.feature_extractor = initialize_feature_extractor()

    def setup_menu(self):
        # 创建菜单栏
        menu_bar = self.menuBar()

        # 创建文件菜单及其动作
        file_menu = menu_bar.addMenu("&File")

        # 创建保存为HTML的动作
        save_html_action = QAction("&Save as HTML", self)
        save_html_action.triggered.connect(self.save_html)
        file_menu.addAction(save_html_action)

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 创建帮助菜单及其动作
        help_menu = menu_bar.addMenu("&Help")
        usage_action = QAction("&Usage Instructions", self)
        usage_action.triggered.connect(self.show_usage_dialog)
        help_menu.addAction(usage_action)
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        # 设置菜单栏样式
        menu_style = """
        QMenuBar {
            background-color: #A8D5D7;
            color: #FFFFFF; /* 菜单栏文字颜色 */
        }

        QMenuBar::item {
            padding: 4px 10px; /* 菜单项内边距 */
        }

        QMenuBar::item:selected {
            background-color: #82B1FF; /* 选中的菜单项背景色，使用亮蓝色 */
        }

        QMenu {
            background-color: #FFFFFF; /* 子菜单背景色，使用白色 */
            color: #333333; /* 子菜单文字颜色 */
        }

        QMenu::item {
            padding: 8px 20px; /* 子菜单项内边距 */
        }

        QMenu::item:selected {
            background-color: #82B1FF; /* 选中的子菜单项背景色，使用亮蓝色 */
        }

        QMenu::separator {
            background-color: #DDDDDD; /* 分隔线颜色 */
            height: 1px; /* 分隔线高度 */
            margin: 6px 0; /* 分隔线外边距 */
        }

        QMenu::indicator {
            width: 13px; /* 子菜单箭头宽度 */
            height: 13px; /* 子菜单箭头高度 */
        }

        QMenu::indicator:unchecked {
            image: url(unchecked.png); /* 未选中状态的箭头图标 */
        }

        QMenu::indicator:checked {
            image: url(checked.png); /* 选中状态的箭头图标 */
        }
        """
        menu_bar.setStyleSheet(menu_style)

    def save_html(self):
        # 获取QTextEdit的内容
        html_content = self.result_area.toHtml()

        # 弹出保存文件对话框
        filename, _ = QFileDialog.getSaveFileName(self, "Save As", "", "HTML Files (*.html)")

        # 如果用户选择了文件路径，则保存
        if filename:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(html_content)
            self.statusBar().showMessage("Saved as HTML with images embedded", 5000)

    def show_usage_dialog(self):
        usage_text = """<h2>Usage Instructions</h2>
        <p>1. Upload a ZIP file containing training images via "File" > "Upload ZIP for Training".</p>
        <p>2. Load your models via "File" > "Upload Models".</p>
        <p>3. Select images for classification via "File" > "Upload Image for Classification".</p>
        <p>4. The application will display the prediction results for each model.</p>"""
        QMessageBox.information(self, "Usage Instructions", usage_text)

    def show_about_dialog(self):
        about_text = """
        <h2>About the CoinVision Project</h2>
        <p><b>CoinVision</b> is an innovative software application designed for the analysis of digitized medieval coin findings. Developed as part of an academic project, it aims to combine the realms of historical numismatics with cutting-edge computer vision and machine learning technologies.</p>

        <h3>Project Goals</h3>
        <p>The primary objectives of the CoinVision project are:</p>
        <ul>
            <li><b>Interdisciplinary Collaboration:</b> Fostering teamwork across disciplines by integrating historical numismatic knowledge with technical expertise in software development.</li>
            <li><b>Advanced Coin Analysis:</b> Utilizing digital representations of coins to explore and identify distinct features and symbols, aiding in the classification and understanding of medieval coinage.</li>
            <li><b>Practical Application of Machine Learning:</b> Applying techniques such as object detection, semantic segmentation, and transfer learning to analyze and categorize coin imagery effectively.</li>
        </ul>

        <h3>Technical Approach</h3>
        <p>The application leverages digitized data from various coin types stored across different archives. Each coin is represented by comprehensive data files including metadata, normal maps, albedo textures, and 3D models with Multi-Scale Integral Invariants (MSII) features.</p>
        <p>Key features of the CoinVision project include:</p>
        <ul>
            <li>Extraction of 2D contours to visualize and recognize motifs on coins.</li>
            <li>Annotation and segmentation of coin features for machine learning model training.</li>
            <li>Division of the dataset into training, testing, and validation subsets to facilitate effective model evaluation and tuning.</li>
        </ul>

        <h3>Collaboration and Development</h3>
        <p>This project is a collaborative effort that emphasizes the importance of project management, responsibility delegation, and clear communication within teams. It utilizes modern software development practices and tools, including version control with git, to ensure a structured and efficient workflow.</p>

        <p>For more detailed information on the CoinVision project, including access to the data and resources, please visit the official project repository or contact the project team.</p>
        """
        QMessageBox.information(self, "About CoinVision", about_text)

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(10)


        # 样式表可以用于改进界面美观性
        self.setStyleSheet("""
            QWidget {
                font-family: 'Arial';  /* 使用 Arial 字体 */
                font-size: 12pt;      /* 设置默认字体大小 */
                color: #333333;       /* 设置默认文字颜色 */
                background-color: #FFFFFF; /* 设置默认背景色 */
            }
            QPushButton {
                font-family: 'Arial';
                font-size: 11pt;      /* 按钮使用稍大字号以突出 */
                color: #FFFFFF;
                background-color: #5F9EA0; /* 按钮背景色，使用深蓝绿色系 */
                border: 1px solid #339;  /* 定义边框为深色 */
                padding: 5px;         /* 按钮内部填充 */
                border-radius: 2px;   /* 圆角边框 */
            }
            QPushButton:hover {
                background-color: #82B1FF; /* 鼠标悬停时的按钮背景色，使用亮蓝色 */
            }
            QLabel {
                font-family: 'Arial';
                font-size: 10pt;      /* 标签使用默认字号 */
                color: #555555;       /* 标签文字颜色稍深 */
            }
            QTextEdit {
                font-family: 'Consolas'; /* 文本编辑框使用等宽字体方便阅读 */
                font-size: 10pt;
                background-color: #EEFFFF; /* 为文本编辑框设置一个淡蓝色背景 */
                color: #005577;       /* 文本编辑框的文字颜色 */
                border: 1px solid #DDDDDD; /* 文本编辑框的边框颜色 */
            }
            QCheckBox {
                font-family: 'Arial';
                font-size: 10pt;      /* 复选框使用默认字号 */
            }
            QSpinBox {
                font-family: 'Arial';
                font-size: 10pt;      /* 微调框使用默认字号 */
                background-color: #DDFFDD; /* 微调框背景色 */
            }
        """)

        # Add the "Upload ZIP for Training" button
        self.upload_zip_btn = QPushButton("Upload ZIP for Training")
        self.upload_zip_btn.clicked.connect(self.upload_zip)
        self.layout.addWidget(self.upload_zip_btn)

        # 创建一个水平布局
        data_augmentation_layout = QHBoxLayout()

        # 创建并添加数据增强复选框到水平布局
        self.data_augmentation_checkbox = QCheckBox("Enable Data Augmentation")
        data_augmentation_layout.addWidget(self.data_augmentation_checkbox)

        # 创建并添加增强计数旋钮到水平布局
        self.augmentation_count_spinbox = QSpinBox()
        self.augmentation_count_spinbox.setMinimum(2)
        self.augmentation_count_spinbox.setMaximum(20)
        self.augmentation_count_spinbox.setValue(2)
        self.augmentation_count_spinbox.setEnabled(False)
        self.augmentation_count_spinbox.setStyleSheet("QSpinBox:disabled { background-color: #DDDDDD; }")
        data_augmentation_layout.addWidget(self.augmentation_count_spinbox)

        # 将水平布局添加到主布局
        self.layout.addLayout(data_augmentation_layout)

        # 复选框的状态更改连接到旋钮的启用/禁用
        self.data_augmentation_checkbox.stateChanged.connect(
            lambda: self.augmentation_count_spinbox.setEnabled(self.data_augmentation_checkbox.isChecked())
        )

        # Add the "Upload Model(s)" button
        self.upload_model_btn = QPushButton("Upload Models")
        self.upload_model_btn.clicked.connect(self.upload_model)
        self.layout.addWidget(self.upload_model_btn)

        # Add the "Upload Image for Classification" button
        self.upload_image_btn = QPushButton("Upload Image for Classification")
        self.upload_image_btn.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_image_btn)

        # Add the progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        # Add the result display area
        self.result_area = CustomTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setPlaceholderText("Results will be displayed here...")
        self.layout.addWidget(self.result_area)

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

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)  # 设置进度条的值
    def start_training(self, zip_file_name, save_path):
        use_data_augmentation = self.data_augmentation_checkbox.isChecked()
        augmentation_count = self.augmentation_count_spinbox.value() if use_data_augmentation else 1
        self.worker = Worker(zip_file_name, save_path, use_data_augmentation, augmentation_count)
        self.worker.update_text.connect(self.result_area.append)  # 用于更新文本区域
        self.worker.update_progress.connect(self.update_progress_bar)  # 确保添加了这一行
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

    def upload_model(self):
        model_filenames, _ = QFileDialog.getOpenFileNames(self, "Select Model Files", "", "Model Files (*.pkl)")
        if len(model_filenames) < 2:
            self.result_area.append("Please select at least two models for comparison.")
            return

        self.models.clear()
        self.model_names.clear()  # 清除之前的模型名称
        for model_filename in model_filenames:
            try:
                model = joblib.load(model_filename)
                self.models.append(model)
                self.model_names.append(os.path.basename(model_filename))  # 保存模型名称
                self.result_area.append(f"Loaded model: {model_filename}")
            except Exception as e:
                self.result_area.append(f"Failed to load model {model_filename}: {e}")

        self.result_area.append(f"{len(self.models)} models loaded successfully for comparison.\n")

    def upload_image(self):
        image_filenames, _ = QFileDialog.getOpenFileNames(self, "Select Images", "",
                                                          "Image Files (*.jpg *.jpeg *.png *.tif)")
        if image_filenames:
            for image_filename in image_filenames:
                self.process_image(image_filename)

    def process_image(self, image_filename):
        # 检查是否已经加载了至少一个模型
        if not self.models:
            self.result_area.append("No models loaded. Please upload a model first.\n")
            return

        try:
            self.progress_bar.setValue(0)  # 开始处理图像时重置进度条
            image = Image.open(image_filename).convert('RGB').resize((224, 224))

            # 检查图像的宽度和高度是否大于零
            if image.width == 0 or image.height == 0:
                raise ValueError("Image dimensions are zero.")

            self.result_area.append(f"Processing image: {os.path.basename(image_filename)}")
            self.progress_bar.setValue(20)  # 图像加载和预处理完成

            # 计算得分百分比
            scores = []
            for model in self.models:
                score = self.predict_image(model, image)
                scores.append(score)
            percentages = self.calculate_percentages(scores)

            # 找出最高置信度的模型的索引
            max_index = np.argmax(percentages)

            # 显示每个模型的结果
            self.result_area.append("Model Comparison Results:")
            for index, (model_name, percentage) in enumerate(zip(self.model_names, percentages)):
                # 检查当前模型是否是置信度最高的模型
                if index == max_index:
                    # 置信度最高的模型使用绿色显示
                    self.result_area.insertHtml(
                        f"<br><span style='color: green;'>Model '{model_name}' has a confidence score of {percentage:.2%}.</span>")
                else:
                    # 其他模型使用默认颜色
                    self.result_area.insertHtml(
                        f"<br><span style='color: red;'>Model '{model_name}' has a confidence score of {percentage:.2%}.</span>")

            # 将 TIFF 图片转换为 base64 编码的 PNG 并插入到 HTML 中
            encoded_png = tiff_to_base64_png(image_filename)
            image_html = f'<br><img src="data:image/png;base64,{encoded_png}" width="200" /><br>'
            self.result_area.insertHtml(image_html)

            self.result_area.append("")  # 添加空行以分隔连续的预测结果
            self.progress_bar.setValue(100)  # 模型对比完成，进度条达到100%
        except Exception as e:
            self.result_area.append(f"Failed to process image {os.path.basename(image_filename)}: {e}\n")

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



class CustomTextEdit(QTextEdit):
        def __init__(self, parent=None):
            super(CustomTextEdit, self).__init__(parent)

        def contextMenuEvent(self, event):
            menu = QMenu(self)

            # 添加复制动作
            copyAction = menu.addAction("Copy")
            copyAction.triggered.connect(self.copy)
            # 只有在有选中文本的时候才启用复制
            copyAction.setEnabled(self.textCursor().hasSelection())

            # 添加全选动作
            selectAllAction = menu.addAction("Select All")
            selectAllAction.triggered.connect(self.selectAll)

            # 添加清除所有记录动作
            clearAction = menu.addAction("Clear All")
            clearAction.triggered.connect(self.clear)

            # 设置菜单样式
            menu_style = """
                    QMenu {
                        background-color: #F5F5F5; /* 设置右键菜单背景色 */
                        color: #333; /* 设置右键菜单文本颜色 */
                    }

                    QMenu::item:selected {
                        background-color: #DDD; /* 设置右键菜单选中项背景色 */
                    }
                    """
            menu.setStyleSheet(menu_style)

            menu.exec(event.globalPos())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = ImageClassifierApp()
    mainWin.show()
    sys.exit(app.exec())

