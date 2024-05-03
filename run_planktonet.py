import sys
sys.path.append('./')
sys.path.append('./src/')
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow, QMainWindow, QVBoxLayout, QLabel, QWidget, QFileDialog, QMessageBox, QScrollArea, QDialog, QVBoxLayout, QInputDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from src import resnet_sort, resnet_finetune, resnet_eval, vit_sort, vit_finetune, vit_eval

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
        pixmap = QPixmap("./img/D20230307T053258_IFCB108_02078.png")
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

        self.resnet_button = QPushButton("Convolutional Network")
        self.resnet_button.clicked.connect(self.openresnetWindow)
        self.central_widget.layout().addWidget(self.resnet_button)

        self.vit_button = QPushButton("Transformer Network")
        self.vit_button.clicked.connect(self.openvitWindow)
        self.central_widget.layout().addWidget(self.vit_button)

        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.initUI)
        self.central_widget.layout().addWidget(self.back_button)

    def openExistingModel(self):
        self.existing_window = ExistingModelWindow()
        self.existing_window.show()

    def openresnetWindow(self):
        self.resnet_window = resnetWindow()
        self.resnet_window.show()

    def openvitWindow(self):
        self.vit_window = vitWindow()
        self.vit_window.show()

    def openEvaluationWindow(self):
        self.evaluation_window = EvaluationWindow()
        self.evaluation_window.show()

    def openFineTuneWindow(self):
        self.fine_tune_window = FineTuneWindow()
        self.fine_tune_window.show()

################################################################################################################################
################################################################################################################################

class resnetWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("resnet Window")
        self.setGeometry(100, 100, 800, 600)
        self.model_file = "./models/pretrained_models/Convolutional/model.pth"
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

        self.resnet_label = QLabel("<h2>Convolutional Neural Network</h2>"
                                "<p><i>Convolutional neural networks are highly effective for image classification tasks.</i>" "<p><b>Select Input Directory:</b> Choose the folder where your images are stored." "<p><b>Select Output Directory:</b> Your newly-sorted photos, as well as some other metrics, will be copied to this folder." "<p><b>Start Sorting:</b> Click this button to begin!"
        )
        self.resnet_label.setWordWrap(True) 
        layout.addWidget(self.resnet_label)

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

            # Call resnet_sort.main() with selected arguments
            resnet_sort.main(self.input_directory, self.model_file, self.output_directory)
            QMessageBox.information(self, "Status Report", "Sorting Complete.", QMessageBox.Ok)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def goBackToMain(self):
        self.close()  # Close the resnetWindow

################################################################################################################################
################################################################################################################################

class vitWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transformer Window")
        self.setGeometry(100, 100, 800, 600)  # Initial window size
        self.model_path = "models/pretrained_models/Transformer/model.pth"
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

        self.vit_label = QLabel("<h2>Transformer</h2>"
                                         "<p><i>A transformer is a type of neural network architecture that leverages self-attention mechanisms to learn contextual relationships between elements in sequential data.</i>" "<p><b>Select Input Directory:</b> Choose the folder where your images are stored." "<p><b>Select Output Directory:</b> Your newly-sorted photos, as well as some other metrics, will be copied to this folder." "<p><b>Start Sorting:</b> Click this button to begin!")

        self.vit_label.setWordWrap(True) 
        layout.addWidget(self.vit_label)
        
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

            # Call vit_sort.main() with selected arguments
            vit_sort.main(self.input_directory, self.model_path, self.output_directory)
            QMessageBox.information(self, "Status Report", "Sorting Complete.", QMessageBox.Ok)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def goBackToMain(self):
        self.close()  # Close the vitWindow

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
        
        # Create a scroll area for the central widget
        scroll_area = QScrollArea()
        widget = QWidget()
        widget.setLayout(layout)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        self.setCentralWidget(scroll_area)

        self.evaluation_label = QLabel("<h2>Model Evaluation</h2>"
                                         "<p><i>Evaluation involves measuring how well a model performs its intended task by comparing the model's predictions to known ground truth labels.</i>" "<p><b>Choose Model Type:</b> Convolutional or Transformer." "<p><b>Select Evaluation Dataset:</b> Choose a set of sorted images to be used as ground truths during evaluation." "<p><b>Select Model File:</b> Choose the model you wish to evaluate." "<p><b>Select Output Directory:</b> Choose where to save the evaluation results." "<p><b>Evaluate:</b> Click this button to begin!")

        layout.addWidget(self.evaluation_label)

        # Button to choose model type
        self.chooseModelTypeButton = QPushButton("Choose Model Type")
        self.chooseModelTypeButton.clicked.connect(self.chooseModelType)
        layout.addWidget(self.chooseModelTypeButton)

        # Button to select evaluation dataset
        self.selectDatasetButton = QPushButton("Select Evaluation Dataset")
        self.selectDatasetButton.clicked.connect(self.selectEvaluationDataset)
        layout.addWidget(self.selectDatasetButton)

        # Button to select model path
        self.selectModelButton = QPushButton("Select Model File")
        self.selectModelButton.clicked.connect(self.selectModelPath)
        layout.addWidget(self.selectModelButton)

        # Button to select output directory
        self.selectOutputButton = QPushButton("Select Output Directory")
        self.selectOutputButton.clicked.connect(self.selectOutputDirectory)
        layout.addWidget(self.selectOutputButton)

        # Buttons for starting evaluation
        self.evaluateButton = QPushButton("Evaluate")
        self.evaluateButton.clicked.connect(self.startEvaluation)
        layout.addWidget(self.evaluateButton)

        # Button to go back to MainWindow
        self.backButton = QPushButton("Close")
        self.backButton.clicked.connect(self.goBackToMain)
        layout.addWidget(self.backButton)

        self.setFixedWidth(800)

        # Attributes to store user selections
        self.model_type = None
        self.evaluation_dataset_path = None
        self.model_path = None
        self.output_directory = None

    def selectModelPath(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", filter="Model files (*.pth)")

    def selectOutputDirectory(self):
        self.output_directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")

    def chooseModelType(self):
        items = ("Convolutional", "Transformer")
        item, ok = QInputDialog.getItem(self, "Choose Model Type", "Select model type:", items, 0, False)
        if ok and item:
            self.model_type = item.lower()
            QMessageBox.information(self, "Model Type Selected", f"Model type selected: {item}")

    def selectEvaluationDataset(self):
        self.evaluation_dataset_path = QFileDialog.getExistingDirectory(self, "Select Evaluation Dataset Directory")

    def startEvaluation(self):
        try:
            # Check if model type is selected
            if not self.model_type:
                QMessageBox.warning(self, "Model Type Not Selected", "Please choose a model type.")
                return

            # Check if evaluation dataset directory is selected
            if not self.evaluation_dataset_path:
                QMessageBox.warning(self, "Evaluation Dataset Not Selected", "Please select the evaluation dataset directory.")
                return

            # Check if output directory is selected
            if not self.output_directory:
                QMessageBox.warning(self, "Output Directory Not Selected", "Please select the output directory.")
                return

            # Check if model file is selected
            if not self.model_path:
                QMessageBox.warning(self, "Model File Not Selected", "Please select the model file.")
                return

            if self.model_type == "convolutional":
                # Call resnet_eval.main() with selected arguments
                resnet_eval.main(self.evaluation_dataset_path, self.model_path, self.output_directory)
                QMessageBox.information(self, "Status Report", "Evaluation Complete.", QMessageBox.Ok)

            elif self.model_type == "transformer":
                # Call vit_eval.main() with selected arguments
                vit_eval.main(self.evaluation_dataset_path, self.model_path, self.output_directory)
                QMessageBox.information(self, "Status Report", "Evaluation Complete.", QMessageBox.Ok)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def goBackToMain(self):
        self.close()  # Close the EvaluationWindow


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

        self.existing_label = QLabel("<h2>Sort with Existing Model</h2>"
                                         "<p><i>This module is designed to sort images from a specified directory into subfolders based on the predictions made by a pre-trained model.</i>" "<p><b>Choose Model Type:</b> Convolutional or Transformer." "<p><b>Select Input Images:</b> Choose a folder containing the images you wish to sort." "<p><b>Select Model File:</b> Choose the model you wish to evaluate." "<p><b>Select Output Directory:</b> Choose where to save the evaluation results." "<p><b>Start Sorting:</b> Click this button to begin!")

        self.existing_label.setWordWrap(True) 
        layout.addWidget(self.existing_label)

        # Button to choose model type
        self.chooseModelTypeButton = QPushButton("Choose Model Type")
        self.chooseModelTypeButton.clicked.connect(self.chooseModelType)
        layout.addWidget(self.chooseModelTypeButton)

        # Buttons for selecting arguments
        self.selectInputButton = QPushButton("Select Input Images")
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

    def selectInputDirectory(self):
        # Prompt user to select input directory
        input_directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if input_directory:
            self.input_directory = input_directory
            QMessageBox.information(self, "Input Directory Selected", f"Input directory selected: {input_directory}")

    def chooseModelType(self):
        items = ("Convolutional", "Transformer")
        item, ok = QInputDialog.getItem(self, "Choose Model Type", "Select model type:", items, 0, False)
        if ok and item:
            self.model_type = item.lower()
            QMessageBox.information(self, "Model Type Selected", f"Model type selected: {item}")

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
                # Call resnet_sort.main() with selected arguments
                resnet_sort.main(self.input_directory, self.model_file, self.output_directory)
                QMessageBox.information(self, "Status Report", "Sorting Complete.", QMessageBox.Ok)

            elif self.model_type == "transformer":
                # Call vit_sort.main() with selected arguments
                vit_sort.main(self.input_directory, self.model_file, self.output_directory)
                QMessageBox.information(self, "Status Report", "Sorting Complete.", QMessageBox.Ok)
                
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
        
        # Create a scroll area for the central widget
        scroll_area = QScrollArea()
        widget = QWidget()
        widget.setLayout(layout)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        self.setCentralWidget(scroll_area)

        self.finetune_label = QLabel("<h2>Fine-Tuning a Model</h2>"
                                         "<p><i>Fine-tuning in machine learning refers to the process of taking a pre-trained model and further training it on new data. This is typically done so the model can perform better on a specific task.</i>" "<p><b>Choose Model Type:</b> Choose the type of model (Convolutional or Transformer) that you wish to fine-tune." "<p><b>Select Base Model:</b> Choose the model you want to fine-tune. This can be one of our base models, or a previously fine-tuned model you wish to refine further." "<p><b>Select Training Data Folder:</b> Choose the folder containing your labeled images. Images should be sorted into sub-folders correctly named with your desired category names. We call these categories <i>classes</i>" "<p><b>Select Output Directory:</b> Choose where you would like to save the updated model and associated training metric files." "<p><b>Choose Learning Rate:</b> We found a learning rate of around 0.0001 produced the best results." "<p><b>Choose Number of Images per Class:</b> We found 300 images per class produced the best results. If you do not have enough images, the program will augment the data available in an attempt to fill in the gap." "<p><b>Start Training:</b> Click this button to begin!")

        self.finetune_label.setWordWrap(True) 
        layout.addWidget(self.finetune_label)

        self.model_type_button = QPushButton("Choose Model Type")
        self.model_type_button.clicked.connect(self.chooseModelType)
        layout.addWidget(self.model_type_button)

        self.base_model_button = QPushButton("Select Base Model")
        self.base_model_button.clicked.connect(self.selectModelFile)
        layout.addWidget(self.base_model_button)

        self.training_data_button = QPushButton("Select Training Data Folder")
        self.training_data_button.clicked.connect(self.selectInputDirectory)
        layout.addWidget(self.training_data_button)

        self.output_directory_button = QPushButton("Select Output Directory")
        self.output_directory_button.clicked.connect(self.selectOutputDirectory)
        layout.addWidget(self.output_directory_button)

        self.learning_rate_button = QPushButton("Choose Learning Rate")
        self.learning_rate_button.clicked.connect(self.chooseLearningRate)
        layout.addWidget(self.learning_rate_button)

        self.num_imgs_button = QPushButton("Choose Number of Images per Class")
        self.num_imgs_button.clicked.connect(self.chooseNumImgs)
        layout.addWidget(self.num_imgs_button)

        self.start_training_button = QPushButton("Start Fine-tuning")
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
        print("Start fine-tuning")
        try:
            # Check if all required arguments are selected
            if not hasattr(self, 'input_directory') or not hasattr(self, 'model_file') or not hasattr(self, 'output_directory') or not hasattr(self, 'model_type') or not hasattr(self, 'lr') or not hasattr(self, 'num_imgs'):
                QMessageBox.warning(self, "Missing Information", "Please make sure to select all required parameters.")
                return

            if self.model_type == "convolutional":
                # Call resnet_finetune.main() with selected arguments
                resnet_finetune.main(self.input_directory, self.output_directory, self.model_file, self.lr, self.num_imgs)
                QMessageBox.information(self, "Status Report", "Fine-tuning Complete.", QMessageBox.Ok)

            if self.model_type == "transformer":
                # Call vit_sort.main() with selected arguments
                vit_finetune.main(self.input_directory, self.output_directory, self.model_file, self.lr, self.num_imgs)
                QMessageBox.information(self, "Status Report", "Fine-tuning Complete.", QMessageBox.Ok)
                
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
