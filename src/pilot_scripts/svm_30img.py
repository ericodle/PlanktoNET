import os
import random
import sys
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def main(data_dir):
    # Define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset and filter out classes with no images
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    classes = dataset.classes

    # Printout of classes
    print(f"Number of classes found in the dataset: {len(classes)}")
    print("Classes found in the dataset:")
    for cls in classes:
        print(cls)

    # Split dataset into train and test sets
    train_set = []
    test_set = []
    for class_idx, class_name in enumerate(classes):
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
    class CustomDataset():
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

    # Extract features from the dataset
    train_features, train_labels = [], []
    for inputs, targets in train_loader:
        train_features.extend(inputs.numpy())
        train_labels.extend(targets.numpy())
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    test_features, test_labels = [], []
    for inputs, targets in test_loader:
        test_features.extend(inputs.numpy())
        test_labels.extend(targets.numpy())
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Scale features
    scaler = StandardScaler()
    train_features_flat = train_features.reshape(train_features.shape[0], -1)
    test_features_flat = test_features.reshape(test_features.shape[0], -1)

    train_features_scaled = scaler.fit_transform(train_features_flat)
    test_features_scaled = scaler.transform(test_features_flat)

    # Train SVM classifier
    svm_classifier = svm.SVC(kernel='sigmoid')
    svm_classifier.fit(train_features_scaled, train_labels)

    # Evaluate SVM classifier
    train_predictions = svm_classifier.predict(train_features_scaled)
    test_predictions = svm_classifier.predict(test_features_scaled)

    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print(f'Training Accuracy: {train_accuracy:.2f}')
    print(f'Test Accuracy: {test_accuracy:.2f}')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    main(data_dir)

