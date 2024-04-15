import os
import sys
import torch
import timm
import numpy as np
from PIL import Image
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

    return actual_labels, predicted_labels, class_names

def load_vit_model(model_path, num_classes):
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_path)
    # Adjust the last layer to match the number of classes in your dataset
    state_dict['head.weight'] = state_dict['head.weight'][:num_classes, :]
    state_dict['head.bias'] = state_dict['head.bias'][:num_classes]
    model.load_state_dict(state_dict, strict=False)
    return model, num_classes

def main(data_dir, model_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Load dataset and create data loader
    dataset = CustomImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Get subfolder names from the dataset
    subfolder_names = sorted(os.listdir(data_dir))

    # Load model and number of classes
    model, num_classes = load_vit_model(model_path, len(subfolder_names))
    model.to(device)
    model.eval()

    # Print class names from dataset and model
    print("Class names from dataset:")
    print(subfolder_names)
    print("\nNumber of classes determined from dataset:", num_classes)

    # Check if number of classes from the model matches the number of classes in the dataset
    if num_classes != len(subfolder_names):
        print("Error: Number of classes from model does not match the number of classes in dataset.")
        print("Please ensure that the number of classes in the model matches the number of classes in the dataset.")
        sys.exit(1)

    # Evaluate model performance
    actual_labels, predicted_labels, class_names = evaluate_sorting_performance(model, test_loader, device, subfolder_names)


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
    conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=subfolder_names)

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

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <test image directory> <path to pre-trained model> <output directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]
    main(data_dir, model_path, output_dir)

