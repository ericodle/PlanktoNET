import os
import random
import sys
import torch
import numpy as np
import torchvision
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main(data_dir, output_dir, learning_rate, num_imgs):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a fixed size
        transforms.CenterCrop(224),  # Crop the center of the image to 224x224
        transforms.ToTensor()
    ])

    # Load dataset and filter out classes with no images
    dataset = datasets.ImageFolder(root=data_dir, transform=None)
    class_names = [class_name for class_name in dataset.classes if len(os.listdir(os.path.join(data_dir, class_name))) > 0]

    # Printout of classes
    print(f"Number of classes found in the dataset: {len(class_names)}")
    print("Classes found in the dataset:")
    for cls in class_names:
        print(cls)

    # Split dataset into train and test sets
    train_set = []
    test_set = []
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        image_paths = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path)]
        if len(image_paths) > num_imgs:
            image_paths = random.sample(image_paths, num_imgs)  # Randomly select desired number of images

        # Apply data augmentation if class has less than required
        if len(image_paths) < num_imgs:
            # Define data augmentation transformations
            augmentation_transform = transforms.Compose([
                transforms.RandomHorizontalFlip()
            ])

            # Calculate number of additional samples needed
            num_additional_samples = num_imgs - len(image_paths)

            # Augment existing images to create additional training samples
            for i in range(num_additional_samples):
                img_path = random.choice(image_paths)
                image = Image.open(img_path).convert("RGB")
                augmented_image = augmentation_transform(image)
                train_set.append((augmented_image, class_idx, class_name))

        # Proceed with the existing code for adding images to train_set and test_set
        train_size = int(0.9 * len(image_paths))
        train_set.extend([(img_path, class_idx, class_name) for img_path in image_paths[:train_size]])
        test_set.extend([(img_path, class_idx, class_name) for img_path in image_paths[train_size:]])

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
            img_path, label, class_name = self.dataset[idx]
            if isinstance(img_path, str):
                image = Image.open(img_path).convert("RGB")
            else:
                image = img_path
            if self.transform:
                image = self.transform(image)
            return image, label, class_name

    # Create custom datasets
    train_dataset = CustomDataset(train_set, transform=transform)
    test_dataset = CustomDataset(test_set, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load pre-trained AlexNet model
    alexnet_model = torchvision.models.alexnet(pretrained=True)
    num_ftrs = alexnet_model.classifier[6].in_features
    alexnet_model.classifier[6] = torch.nn.Linear(num_ftrs, len(class_names))  # Change the last fully connected layer to match the number of classes

    # Move model to device
    alexnet_model = alexnet_model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(alexnet_model.parameters(), lr=learning_rate) 

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)

    # Log file
    log_file_path = os.path.join(output_dir, 'training_log.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch\tTrain Loss\tTest Loss\tTest Accuracy\n")

        # Training loop
        train_losses = []
        test_losses = []
        test_accuracies = []

        num_epochs = 1000000 # Set arbitrarily large
        best_test_accuracy = 0.0
        epochs_since_improvement = 0
        stop_training = False

        # Training loop
        for epoch in range(num_epochs):
            alexnet_model.train()
            running_loss = 0.0
            for batch_idx, (inputs, labels, class_names) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = alexnet_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

                # Save transformed images
                for i in range(inputs.size(0)):
                    class_output_dir = os.path.join(output_dir, class_names[i])
                    if not os.path.exists(class_output_dir):
                        os.makedirs(class_output_dir)
                    output_path = os.path.join(class_output_dir, f'image_{epoch}_{batch_idx}_{i}.jpg')
                    torchvision.utils.save_image(inputs[i], output_path)

            epoch_loss = running_loss / len(train_set)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')
            train_losses.append(epoch_loss)

            # Update the learning rate scheduler after optimizer's step
            scheduler.step()

            # Testing
            alexnet_model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = alexnet_model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            test_loss /= len(test_set)
            test_accuracy = correct / total
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}')
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            # Write to log file
            log_file.write(f"{epoch+1}\t{epoch_loss:.4f}\t{test_loss:.4f}\t{test_accuracy:.2%}\n")

            # Check for early stopping
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= 2:
                # Save trained model
                model_save_path = os.path.join(output_dir, 'alexnet_model.pth')
                torch.save(alexnet_model.state_dict(), model_save_path)
                print(f"Trained model saved at: {model_save_path}")  

                print("Early stopping triggered. Test accuracy did not improve within patience epochs.")
                stop_training = True
                break

            if stop_training:       
                break

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Training and Testing Metrics')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'training_plot.png'))
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <train/test image directory> <output directory> <initial learning rate> <images per class>")
        sys.exit(1)
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    learning_rate = float(sys.argv[3])
    num_imgs = int(sys.argv[4])
    main(data_dir, output_dir, learning_rate, num_imgs)

