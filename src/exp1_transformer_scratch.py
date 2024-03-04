import os
import random
import sys
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeiTModel, AdamW, get_linear_schedule_with_warmup

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset[idx]
        if isinstance(img_path, str):
            image = Image.open(img_path).convert("RGB")
        else:
            image = img_path
        if self.transform:
            image = self.transform(image)
        return image, label

def main(data_dir, output_dir, learning_rate):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load dataset and filter out classes with no images
    dataset = datasets.ImageFolder(root=data_dir, transform=None)
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
        if len(image_paths) > 100:
            image_paths = random.sample(image_paths, 30)  # Randomly select 30 images
        
        # Apply data augmentation if class has less than 30 images
        if len(image_paths) < 100:
            # Define data augmentation transformations
            augmentation_transform = transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])

            # Calculate number of additional samples needed
            num_additional_samples = 30 - len(image_paths)

            # Augment existing images to create additional training samples
            for i in range(num_additional_samples):
                img_path = random.choice(image_paths)
                image = Image.open(img_path).convert("RGB")
                augmented_image = augmentation_transform(image)
                train_set.append((augmented_image, class_idx))
        
        # Proceed with the existing code for adding images to train_set and test_set
        train_size = int(0.9 * len(image_paths))
        train_set.extend([(img_path, class_idx) for img_path in image_paths[:train_size]])
        test_set.extend([(img_path, class_idx) for img_path in image_paths[train_size:]])

    # Shuffle train and test sets
    random.shuffle(train_set)
    random.shuffle(test_set)

    # Define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom datasets
    train_dataset = CustomDataset(train_set, transform=transform)
    test_dataset = CustomDataset(test_set, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224').to(device)
    model.train()

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader))

    # Training loop
    best_test_accuracy = 0.0
    patience = 3
    epochs_since_improvement = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            logits = outputs.last_hidden_state[:, 0]  # Extract CLS token embedding for classification
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                logits = outputs.last_hidden_state[:, 0]
                loss = F.cross_entropy(logits, labels)
                test_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = correct / total
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Check for early stopping
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered. Test accuracy did not improve within {patience} epochs.")
            break

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <data_directory> <output_directory> <learning_rate>")
        sys.exit(1)
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    learning_rate = float(sys.argv[3])
    num_epochs = 10  # Set the number of epochs
    main(data_dir, output_dir, learning_rate)
