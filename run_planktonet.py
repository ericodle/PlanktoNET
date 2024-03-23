import sys
sys.path.append('./')
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow, QMainWindow, QVBoxLayout, QLabel, QWidget, QFileDialog, QMessageBox, QScrollArea, QDialog, QVBoxLayout, QInputDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from src import cnn_sort, cnn_finetune, cnn_eval, transformer_sort, transformer_finetune, transformer_eval

################################################################################################################################
################################################################################################################################

class StartWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        title_label = QLabel("<h1>PlanktoNet: Plankton Image Sorter</h1>")
        layout.addWidget(title_label)

        # Add a cute image
        image_label = QLabel(alignment=Qt.AlignCenter) 
        pixmap = QPixmap("/home/eo/Desktop/PlanktoNET-main/img/D20230307T053258_IFCB108_02078.png")
        pixmap = pixmap.scaledToWidth(400)  # Adjust the width as needed
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)

        release_label = QLabel("Release Information: Version 1.0")
        layout.addWidget(release_label)

        authors_label = QLabel("Authors: Eric Odle and Phua")
        layout.addWidget(authors_label)

        begin_button = QPushButton("Begin")
        begin_button.clicked.connect(self.openMainWindow)
        layout.addWidget(begin_button)

        self.setLayout(layout)

    def openMainWindow(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()

################################################################################################################################
################################################################################################################################

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Main Window")
        self.resize(400, 300)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        self.sort_using_new_model_button = QPushButton("Sort with New Model") 
        self.sort_using_new_model_button.clicked.connect(self.showChooseModel) 
        layout.addWidget(self.sort_using_new_model_button) 

        self.use_existing_model_button = QPushButton("Sort with Existing Model")
        self.use_existing_model_button.clicked.connect(self.openExistingModel)
        layout.addWidget(self.use_existing_model_button)

        self.fine_tune_model_button = QPushButton("Fine-Tune Model") 
        self.fine_tune_model_button.clicked.connect(self.openFineTuneWindow) 
        layout.addWidget(self.fine_tune_model_button) 

        self.evaluate_model_button = QPushButton("Evaluate Model")
        self.evaluate_model_button.clicked.connect(self.openEvaluationWindow)
        layout.addWidget(self.evaluate_model_button)

        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(self.close)
        layout.addWidget(quit_button)

        self.central_widget.setLayout(layout)

    def showChooseModel(self):
        for i in reversed(range(self.central_widget.layout().count())):
            widget = self.central_widget.layout().itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        self.model_selection_label = QLabel("<h1>Choose Model</h1>")
        self.central_widget.layout().addWidget(self.model_selection_label)

        self.cnn_button = QPushButton("Convolutional Neural Network")
        self.cnn_button.clicked.connect(self.openCNNWindow)
        self.central_widget.layout().addWidget(self.cnn_button)

        self.transformer_button = QPushButton("Transformer Neural Network")
        self.transformer_button.clicked.connect(self.openTransformerWindow)
        self.central_widget.layout().addWidget(self.transformer_button)

        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.initUI)
        self.central_widget.layout().addWidget(self.back_button)

    def openExistingModel(self):
        self.existing_window = ExistingModelWindow()
        self.existing_window.show()

    def openCNNWindow(self):
        self.cnn_window = CNNWindow()
        self.cnn_window.show()

    def openTransformerWindow(self):
        self.transformer_window = TransformerWindow()
        self.transformer_window.show()

    def openEvaluationWindow(self):
        self.evaluation_window = EvaluationWindow()
        self.evaluation_window.show()

    def openFineTuneWindow(self):
        self.fine_tune_window = FineTuneWindow()
        self.fine_tune_window.show()

################################################################################################################################
################################################################################################################################

class CNNWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Window")
        self.setGeometry(100, 100, 800, 600)
        self.model_file = "./models/pretrained_models/cnn_model.pth"
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Create a scroll area for the central widget
        scroll_area = QScrollArea()
        widget = QWidget()
        widget.setLayout(layout)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        self.setCentralWidget(scroll_area)

        self.cnn_label = QLabel("<h2>Convolutional Neural Network (CNN)</h2>"
                                "<p>A Convolutional Neural Network (CNN) is a type of artificial neural network "
                                "that is highly effective for image classification tasks. It consists of multiple "
                                "layers, including convolutional layers, pooling layers, and fully connected layers. "
                                "CNNs are designed to automatically and adaptively learn spatial hierarchies of features "
                                "from input images.</p>"
                                "<p>CNNs are particularly advantageous in sorting plankton images because they can "
                                "capture intricate patterns and features at different levels of abstraction, allowing "
                                "for accurate classification even in complex and varied environments.</p>"
                                "<p>Using pre-trained CNN models, such as ResNet101, can further enhance performance "
                                "by leveraging knowledge learned from large datasets in related domains.</p>")
        self.cnn_label.setWordWrap(True) 
        layout.addWidget(self.cnn_label)

        # Buttons for selecting arguments
        self.selectInputButton = QPushButton("Select Input Directory")
        self.selectInputButton.clicked.connect(self.selectInputDirectory)
        layout.addWidget(self.selectInputButton)

        self.selectOutputButton = QPushButton("Select Output Directory")
        self.selectOutputButton.clicked.connect(self.selectOutputDirectory)
        layout.addWidget(self.selectOutputButton)

        # Button to start sorting
        self.startSortingButton = QPushButton("Start Sorting")
        self.startSortingButton.clicked.connect(self.startSorting)
        layout.addWidget(self.startSortingButton)

        # Button to go back to MainWindow
        self.backButton = QPushButton("Close")
        self.backButton.clicked.connect(self.goBackToMain)
        layout.addWidget(self.backButton)

        self.setFixedWidth(800)

    def selectInputDirectory(self):
        # Prompt user to select input directory
        input_directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if input_directory:
            self.input_directory = input_directory
            QMessageBox.information(self, "Input Directory Selected", f"Input directory selected: {input_directory}")

    def selectOutputDirectory(self):
        # Prompt user to select output directory
        output_directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if output_directory:
            self.output_directory = output_directory
            QMessageBox.information(self, "Output Directory Selected", f"Output directory selected: {output_directory}")

    def startSorting(self):
        try:
            # Check if all required arguments are selected
            if not hasattr(self, 'input_directory') or not hasattr(self, 'output_directory'):
                QMessageBox.warning(self, "Missing Information", "Please select input images and output directory.")
                return

            # Call cnn_sort.main() with selected arguments
            cnn_sort.main(self.input_directory, self.model_file, self.output_directory)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def goBackToMain(self):
        self.close()  # Close the CNNWindow

################################################################################################################################
################################################################################################################################

class TransformerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transformer Window")
        self.setGeometry(100, 100, 800, 600)  # Initial window size
        self.model_file = "./models/pretrained_models/transformer_model.pth"
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Create a scroll area for the central widget
        scroll_area = QScrollArea()
        widget = QWidget()
        widget.setLayout(layout)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        self.setCentralWidget(scroll_area)

        self.transformer_label = QLabel("<h2>Transformer Neural Network</h2>"
                                         "<p>A Transformer is a type of neural network architecture "
                                         "that is highly effective for sequence-to-sequence tasks, "
                                         "such as natural language processing and image classification. "
                                         "It uses self-attention mechanisms to capture dependencies "
                                         "between different elements in the input data.</p>"
                                         "<p>Transformers have shown promising results in various tasks "
                                         "due to their ability to model long-range dependencies and "
                                         "capture complex patterns in data.</p>"
                                         "<p>For sorting plankton images, Transformer models can analyze "
                                         "the spatial relationships between pixels and capture contextual "
                                         "information, leading to improved classification accuracy.</p>")

        self.transformer_label.setWordWrap(True) 
        layout.addWidget(self.transformer_label)
        
        # Buttons for selecting arguments
        self.selectInputButton = QPushButton("Select Input Directory")
        self.selectInputButton.clicked.connect(self.selectInputDirectory)
        layout.addWidget(self.selectInputButton)

        self.selectOutputButton = QPushButton("Select Output Directory")
        self.selectOutputButton.clicked.connect(self.selectOutputDirectory)
        layout.addWidget(self.selectOutputButton)

        # Button to start sorting
        self.startSortingButton = QPushButton("Start Sorting")
        self.startSortingButton.clicked.connect(self.startSorting)
        layout.addWidget(self.startSortingButton)

        # Button to go back to MainWindow
        self.backButton = QPushButton("Close")
        self.backButton.clicked.connect(self.goBackToMain)
        layout.addWidget(self.backButton)

        self.setFixedWidth(800)

    def selectInputDirectory(self):
        # Prompt user to select input directory
        input_directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if input_directory:
            self.input_directory = input_directory
            QMessageBox.information(self, "Input Directory Selected", f"Input directory selected: {input_directory}")

    def selectOutputDirectory(self):
        # Prompt user to select output directory
        output_directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if output_directory:
            self.output_directory = output_directory
            QMessageBox.information(self, "Output Directory Selected", f"Output directory selected: {output_directory}")

    def startSorting(self):
        try:
            # Check if all required arguments are selected
            if not hasattr(self, 'input_directory') or not hasattr(self, 'output_directory'):
                QMessageBox.warning(self, "Missing Information", "Please select input images and output directory.")
                return

            # Call transformer_sort.main() with selected arguments
            transformer_sort.main(self.input_directory, self.model_file, self.output_directory)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def goBackToMain(self):
        self.close()  # Close the TransformerWindow

################################################################################################################################
################################################################################################################################

class EvaluationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Evaluation Window")
        self.setGeometry(100, 100, 800, 600)  # Initial window size
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Add evaluation functionality here
        
        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)

################################################################################################################################
################################################################################################################################

class ExistingModelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Existing Model Window")
        self.setGeometry(100, 100, 800, 600)  # Initial window size
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Create a scroll area for the central widget
        scroll_area = QScrollArea()
        widget = QWidget()
        widget.setLayout(layout)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        self.setCentralWidget(scroll_area)

        self.existing_label = QLabel("Select an existing model file that you have been working on."
                                  "This allows you to reuse trained models for sorting plankton images.")
        self.existing_label.setWordWrap(True) 
        layout.addWidget(self.existing_label)

        # Button to choose model type
        self.chooseModelTypeButton = QPushButton("Choose Model Type")
        self.chooseModelTypeButton.clicked.connect(self.chooseModelType)
        layout.addWidget(self.chooseModelTypeButton)

        # Buttons for selecting arguments
        self.selectInputButton = QPushButton("Select Input Directory")
        self.selectInputButton.clicked.connect(self.selectInputDirectory)
        layout.addWidget(self.selectInputButton)

        self.selectModelButton = QPushButton("Select Model File")
        self.selectModelButton.clicked.connect(self.selectModelFile)
        layout.addWidget(self.selectModelButton)

        self.selectOutputButton = QPushButton("Select Output Directory")
        self.selectOutputButton.clicked.connect(self.selectOutputDirectory)
        layout.addWidget(self.selectOutputButton)

        # Button to start sorting
        self.startSortingButton = QPushButton("Start Sorting")
        self.startSortingButton.clicked.connect(self.startSorting)
        layout.addWidget(self.startSortingButton)

        # Button to go back to MainWindow
        self.backButton = QPushButton("Close")
        self.backButton.clicked.connect(self.goBackToMain)
        layout.addWidget(self.backButton)

        self.setFixedWidth(800)

    def chooseModelType(self):
        items = ("Convolutional", "Transformer")
        item, ok = QInputDialog.getItem(self, "Choose Model Type", "Select model type:", items, 0, False)
        if ok and item:
            self.model_type = item.lower()
            QMessageBox.information(self, "Model Type Selected", f"Model type selected: {item}")

    def selectInputDirectory(self):
        # Prompt user to select input directory
        input_directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if input_directory:
            self.input_directory = input_directory
            QMessageBox.information(self, "Input Directory Selected", f"Input directory selected: {input_directory}")

    def selectModelFile(self):
        # Prompt user to select model file
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", filter="Model files (*.pth)")
        if model_file:
            self.model_file = model_file
            QMessageBox.information(self, "Model File Selected", f"Model file selected: {model_file}")

    def selectOutputDirectory(self):
        # Prompt user to select output directory
        output_directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if output_directory:
            self.output_directory = output_directory
            QMessageBox.information(self, "Output Directory Selected", f"Output directory selected: {output_directory}")

    def startSorting(self):
        try:
            # Check if all required arguments are selected
            if not hasattr(self, 'input_directory') or not hasattr(self, 'model_file') or not hasattr(self, 'output_directory') or not hasattr(self, 'model_type'):
                QMessageBox.warning(self, "Missing Information", "Please select input directory, model file, output directory, and model type.")
                return

            if self.model_type == "convolutional":
                # Call cnn_sort.main() with selected arguments
                cnn_sort.main(self.input_directory, self.model_file, self.output_directory)
            elif self.model_type == "transformer":
                # Call transformer_sort.main() with selected arguments
                transformer_sort.main(self.input_directory, self.model_file, self.output_directory)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def goBackToMain(self):
        self.close()


################################################################################################################################
################################################################################################################################

class FineTuneWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fine-Tune Model")
        self.setGeometry(100, 100, 800, 600)  # Initial window size
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.base_model_button = QPushButton("Select Base Model")
        self.base_model_button.clicked.connect(self.selectModelFile)
        layout.addWidget(self.base_model_button)

        self.training_data_button = QPushButton("Select Training Data Folder")
        self.training_data_button.clicked.connect(self.selectInputDirectory)
        layout.addWidget(self.training_data_button)

        self.output_directory_button = QPushButton("Select Output Directory")
        self.output_directory_button.clicked.connect(self.selectOutputDirectory)
        layout.addWidget(self.output_directory_button)

        self.model_type_button = QPushButton("Choose Model Type")
        self.model_type_button.clicked.connect(self.chooseModelType)
        layout.addWidget(self.model_type_button)

        self.learning_rate_button = QPushButton("Choose Learning Rate")
        self.learning_rate_button.clicked.connect(self.chooseLearningRate)
        layout.addWidget(self.learning_rate_button)

        self.num_imgs_button = QPushButton("Choose Number of Images per Class")
        self.num_imgs_button.clicked.connect(self.chooseNumImgs)
        layout.addWidget(self.num_imgs_button)

        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self.startFinetuning)
        layout.addWidget(self.start_training_button)

        self.back_button = QPushButton("Close")
        self.back_button.clicked.connect(self.goBackToMain)
        layout.addWidget(self.back_button)

        self.setFixedWidth(800)
        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)

    def chooseModelType(self):
        items = ("Convolutional", "Transformer")
        item, ok = QInputDialog.getItem(self, "Choose Model Type", "Select model type:", items, 0, False)
        if ok and item:
            self.model_type = item.lower()
            QMessageBox.information(self, "Model Type Selected", f"Model type selected: {item}")

    def selectInputDirectory(self):
        # Prompt user to select input directory
        input_directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if input_directory:
            self.input_directory = input_directory
            QMessageBox.information(self, "Input Directory Selected", f"Input directory selected: {input_directory}")

    def selectModelFile(self):
        # Prompt user to select model file
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", filter="Model files (*.pth)")
        if model_file:
            self.model_file = model_file
            QMessageBox.information(self, "Model File Selected", f"Model file selected: {model_file}")

    def selectOutputDirectory(self):
        # Prompt user to select output directory
        output_directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if output_directory:
            self.output_directory = output_directory
            QMessageBox.information(self, "Output Directory Selected", f"Output directory selected: {output_directory}")

    def chooseLearningRate(self):
        lr, ok = QInputDialog.getDouble(self, "Choose Learning Rate", "Enter Learning Rate:", decimals=5)
        if ok:
            self.lr = lr
            QMessageBox.information(self, "Learning Rate Selected", f"Learning Rate selected: {lr}")

    def chooseNumImgs(self):
        num_imgs, ok = QInputDialog.getInt(self, "Choose Number of Images per Class", "Enter Number of Images per Class:")
        if ok:
            self.num_imgs = num_imgs
            QMessageBox.information(self, "Number of Images Selected", f"Number of Images Selected: {num_imgs}")

    def startFinetuning(self):
        try:
            # Check if all required arguments are selected
            if not hasattr(self, 'input_directory') or not hasattr(self, 'model_file') or not hasattr(self, 'output_directory') or not hasattr(self, 'model_type') or not hasattr(self, 'lr') or not hasattr(self, 'num_imgs'):
                QMessageBox.warning(self, "Missing Information", "Please make sure to select all required parameters.")
                return

            if self.model_type == "convolutional":
                # Call cnn_sort.main() with selected arguments
                cnn_finetune.main(self.input_directory, self.model_file, self.output_directory, self.lr, self.num_imgs)
            elif self.model_type == "transformer":
                # Call transformer_sort.main() with selected arguments
                transformer_finetune.main(self.input_directory, self.model_file, self.output_directory, self.lr, self.num_imgs)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def goBackToMain(self):
        self.close()

################################################################################################################################
################################################################################################################################

if __name__ == '__main__':
    app = QApplication(sys.argv)
    start_window = StartWindow()
    start_window.show()
    sys.exit(app.exec_())
