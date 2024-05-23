import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import timm 

'''
This script evaluates ViT model performance on a custom image dataset using various evaluation metrics. It loads the model, computes metrics such as accuracy, precision, recall, and F1 score, generates a confusion matrix, and plots ROC curves for each class. Images are then sorted into subfolders based on their predicted classes, and all evaluation results are saved to the specified output directory.
'''

class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Get class directories and assign numerical labels
        for idx, subdir in enumerate(os.listdir(root)):
            subdir_path = os.path.join(root, subdir)
            if os.path.isdir(subdir_path):
                self.class_to_idx[subdir] = idx
                self.idx_to_class[idx] = subdir
                for filename in os.listdir(subdir_path):
                    filepath = os.path.join(subdir_path, filename)
                    # Attempt to open the image, skip if it fails
                    try:
                        Image.open(filepath).verify()
                        self.samples.append((filepath, subdir))
                    except (IOError, SyntaxError) as e:
                        print(f"Skipping {filepath}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        # Convert target (class label) to numerical format
        target = self.class_to_idx[target]

        return sample, target

def evaluate_sorting_performance(model, test_loader, device, class_names):
    predicted_labels = []
    actual_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend([class_names[idx] for idx in predicted.cpu().tolist()])
            actual_labels.extend([test_loader.dataset.idx_to_class[label.item()] for label in labels.cpu()])

    return actual_labels, predicted_labels


def load_vit_model(model_path):
    class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.txt')
    with open(class_names_path, 'r') as f:
        class_names = f.read().splitlines()

    num_classes = len(class_names)

    # Load the Vision Transformer model using timm library
    model = timm.create_model('vit_base_patch16_224', pretrained=False)  # Example: ViT Base model with 16x16 patches and 224x224 image size

    # Replace the classification head with a new one for your specific number of classes
    model.head = torch.nn.Linear(model.head.in_features, num_classes)

    # Load the entire checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Load the model's parameters directly from the checkpoint
    model.load_state_dict(checkpoint)

    return model, class_names


def plot_roc_curve(fpr, tpr, auc, class_names, output_dir, strategy):
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({strategy})')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f'roc_curve_{strategy}.png'))
    plt.close()

def evaluate_roc(model, test_loader, device, class_names, output_dir):
    all_actual_labels = []
    all_predicted_probs = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_probs = torch.softmax(outputs, dim=1)
            all_actual_labels.extend(labels.cpu().numpy())
            all_predicted_probs.extend(predicted_probs.cpu().numpy())

    all_actual_labels = np.array(all_actual_labels)
    all_predicted_probs = np.array(all_predicted_probs)

    # Initialize dictionaries to store FPR, TPR, and AUC for each class
    fpr_class = dict()
    tpr_class = dict()
    roc_auc_class = dict()

    # Calculate ROC curve and AUC for each class
    for i in range(len(class_names)):
        fpr_class[i], tpr_class[i], _ = roc_curve(all_actual_labels == i, all_predicted_probs[:, i])
        roc_auc_class[i] = roc_auc_score(all_actual_labels == i, all_predicted_probs[:, i])

        # Plot ROC curve for each class
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_class[i], tpr_class[i], label=f'{class_names[i]} (AUC = {roc_auc_class[i]:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({class_names[i]})')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, f'roc_curve_{class_names[i]}.png'))
        plt.close()

def main(data_dir, model_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Load model and class names
    model, class_names = load_vit_model(model_path)
    model.to(device)
    model.eval()

    # Get subfolder names from the dataset
    subfolder_names = sorted(os.listdir(data_dir))

    # Check for mismatched class names
    mismatched_classes = set(subfolder_names) ^ set(class_names)
    if mismatched_classes:
        print("Error: Class names from model do not match subfolder names in dataset.")
        print("Please ensure that the classes in the model and dataset match exactly.")
        sys.exit(1)

    # Load dataset and create data loader
    dataset = CustomImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Evaluate model performance
    actual_labels, predicted_labels = evaluate_sorting_performance(model, test_loader, device, class_names)

    # Save sorted images into subfolders with their respective class names in the output directory
    for i, (image_path, label) in enumerate(dataset.samples):
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        image = Image.open(image_path)
        image_name = os.path.basename(image_path)
        image.save(os.path.join(label_dir, image_name))

    # Calculate evaluation metrics
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, average='macro')
    recall = recall_score(actual_labels, predicted_labels, average='macro')
    f1 = f1_score(actual_labels, predicted_labels, average='macro')

    # Generate confusion matrix
    conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=class_names)

    # Print evaluation metrics
    print(f"\nAccuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Save evaluation metrics to a text file
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")

    # Save confusion matrix as an image
    num_classes = len(class_names)
    figsize = (max(10, num_classes * 0.5), max(8, num_classes * 0.4))
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=120)
    plt.close()

    # Evaluate ROC and AUC
    evaluate_roc(model, test_loader, device, class_names, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <test image directory> <path to pre-trained model> <output directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]
    main(data_dir, model_path, output_dir)
