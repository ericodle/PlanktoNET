import os
import sys
import torch
import numpy as np
from PIL import Image
from scipy.stats import ks_2samp
import torchvision
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

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
        sample = Image.open(path).convert('L')

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

def load_resnet_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.txt')
    with open(class_names_path, 'r') as f:
        class_names = f.read().splitlines()

    num_classes = len(class_names)

    resnet_model = torchvision.models.resnet50(weights=None)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = torch.nn.Linear(num_ftrs, num_classes)
    resnet_model.load_state_dict(torch.load(model_path))

    return resnet_model, class_names

def calculate_ks_statistics(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    ks_values = {}
    for i, class_name in enumerate(test_loader.dataset.idx_to_class.values()):
        # True labels for the current class
        true_labels = (all_labels == i).astype(int)
        # Predicted probabilities for the current class
        pred_probs = all_preds[:, i]

        # KS statistic
        ks_stat, _ = ks_2samp(true_labels, pred_probs)
        ks_values[class_name] = ks_stat

    return ks_values

def plot_ks_statistics(ks_values, output_dir):
    num_classes = len(ks_values)
    width = (num_classes)  # Dynamic width based on number of classes
    height = 12

    plt.figure(figsize=(width, height))
    class_names = list(ks_values.keys())
    ks_stats = list(ks_values.values())

    plt.bar(class_names, ks_stats, color='skyblue')
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('KS Statistic', fontsize=20)
    plt.title('KS Statistic for Each Class', fontsize=16)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ks_statistics.png'), dpi=120)  # Higher DPI for better quality
    plt.close()

def main(data_dir, model_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load model and class names
    model, class_names = load_resnet_model(model_path)
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
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=600)
    plt.close()

    # Calculate and plot KS statistics
    ks_values = calculate_ks_statistics(model, test_loader, device)
    plot_ks_statistics(ks_values, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <test image directory> <path to pre-trained model> <output directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]
    main(data_dir, model_path, output_dir)
