import os
import random
import sys
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main(data_dir, output_dir, learning_rate):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.RandomResizedCrop(299),  # Inception network requires input size of 299x299
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
            image_paths = random.sample(image_paths, 100)  # Randomly select 100 images
        
        # Apply data augmentation if class has less than 30 images
        if len(image_paths) < 100:
            # Define data augmentation transformations
            augmentation_transform = transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),  # Inception network input size
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

    # Custom dataset class
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

    # Create custom datasets
    train_dataset = CustomDataset(train_set, transform=transform)
    test_dataset = CustomDataset(test_set, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load pre-trained Inception network
    inception = torchvision.models.inception_v3(pretrained=True)

    # Modify the classifier
    num_features = inception.fc.in_features
    inception.fc = torch.nn.Linear(num_features, len(testid_classes))

    # Move model to device
    inception.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(inception.parameters(), lr=learning_rate) 

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Log file
    log_file_path = os.path.join(output_dir, 'training_log.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch\tTrain Loss\tTest Loss\tTest Accuracy\n")

        # Training loop
        num_epochs = 10000
        best_test_accuracy = 0.0
        epochs_since_improvement = 0
        stop_training = False

        # Training loop
        for epoch in range(num_epochs):
            inception.train()
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = inception(inputs).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            epoch_loss = running_loss / len(train_set)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')

            # Update the learning rate scheduler after optimizer's step
            scheduler.step()

            # Testing
            inception.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = inception(inputs)  # Access logits directly
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(probabilities, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()


            test_loss /= len(test_set)
            test_accuracy = correct / total
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}')

            # Write to log file
            log_file.write(f"{epoch+1}\t{epoch_loss:.4f}\t{test_loss:.4f}\t{test_accuracy:.2%}\n")

            # Check for early stopping
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= 3:
                print("Early stopping triggered. Test accuracy did not improve within 3 epochs.")
                stop_training = True
                break

            if stop_training:            
                break

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <data_directory> <output_directory> <learning_rate>")
        sys.exit(1)
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    learning_rate = float(sys.argv[3])  # Convert learning_rate to float
    main(data_dir, output_dir, learning_rate)

