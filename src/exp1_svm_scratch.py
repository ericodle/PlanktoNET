import os
import random
import sys
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import datasets, transforms, models
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


# Function to extract features from a dataset using a model
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define device here
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            features_batch = model(inputs).cpu().numpy()
            features.append(features_batch)
            labels.append(targets.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

def main(data_dir, output_dir):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset and filter out classes with no images
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    testid_classes = [class_name for class_name in dataset.classes if len(os.listdir(os.path.join(data_dir, class_name))) > 0]

    # Printout of classes
    print(f"Number of classes found in the dataset: {len(testid_classes)}")
    print("Classes found in the dataset:")
    for cls in testid_classes:
        print(cls)

    # Split dataset into train and test sets
    train_set = []
    test_set = []
    for class_idx, class_name in enumerate(testid_classes):
        class_path = os.path.join(data_dir, class_name)
        image_paths = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path)]
        if len(image_paths) > 30:
            image_paths = random.sample(image_paths, 30)  # Randomly select 30 images
        
        train_size = int(0.9 * len(image_paths))
        train_set.extend([(img_path, class_idx) for img_path in image_paths[:train_size]])
        test_set.extend([(img_path, class_idx) for img_path in image_paths[train_size:]])

    # Shuffle train and test sets
    random.shuffle(train_set)
    random.shuffle(test_set)

    # Custom dataset class
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img_path, label = self.dataset[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    # Create custom datasets
    train_dataset = CustomDataset(train_set, transform=transform)
    test_dataset = CustomDataset(test_set, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load pre-trained ResNet model
    resnet_model = models.resnet18(pretrained=True)
    resnet_model = resnet_model.to(device)

    # Remove the last fully connected layer (the classification layer)
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])

    # Extract features from the dataset using ResNet
    train_features, train_labels = extract_features(resnet_model, train_loader)
    test_features, test_labels = extract_features(resnet_model, test_loader)

    # Scale features
    scaler = StandardScaler()
    train_features_flat = train_features.reshape(train_features.shape[0], -1)
    test_features_flat = test_features.reshape(test_features.shape[0], -1)

    train_features_scaled = scaler.fit_transform(train_features_flat)
    test_features_scaled = scaler.transform(test_features_flat)

    # Train SVM classifier
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(train_features_scaled, train_labels)

    # Evaluate SVM classifier
    train_predictions = svm_classifier.predict(train_features_scaled)
    test_predictions = svm_classifier.predict(test_features_scaled)

    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print(f'Training Accuracy: {train_accuracy:.2f}')
    print(f'Test Accuracy: {test_accuracy:.2f}')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <data_directory> <output_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(data_dir, output_dir)

