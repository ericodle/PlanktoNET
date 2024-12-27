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
import torch.nn.functional as F
import timm

class_names_file = "class_names.txt"  # Define the file name for class names

def main(data_dir, output_dir, learning_rate=0.001, num_imgs=100):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(30), 
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset and filter out classes with no images
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = [class_name for class_name in dataset.classes if len(os.listdir(os.path.join(data_dir, class_name))) > 0]

    # Log and store class names
    with open(os.path.join(output_dir, class_names_file), 'w') as f:
        f.write('\n'.join(class_names))

    # Printout of classes
    print(f"Number of classes found in the dataset: {len(class_names)}")
    print("Classes found in the dataset:")
    for cls in class_names:
        print(cls)

    # Function to check if an image file is corrupted
    def is_corrupted_image(file_path):
        try:
            Image.open(file_path).verify()
            return False
        except (IOError, SyntaxError):
            return True

    # Split dataset into train, validation
    train_set = []
    val_set = []
    corrupted_images = []

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        image_paths = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path)]

        # Pre-screen training images for corruption
        filtered_image_paths = []
        for img_path in image_paths:
            if not is_corrupted_image(img_path):
                filtered_image_paths.append(img_path)
            else:
                corrupted_images.append(img_path)

        # If there are fewer images than `num_imgs`, sample with replacement
        if len(filtered_image_paths) < num_imgs:
            filtered_image_paths = random.choices(filtered_image_paths, k=num_imgs)
        else:
            # Otherwise, randomly sample a fixed number of images
            filtered_image_paths = random.sample(filtered_image_paths, num_imgs)

        # Split into train, validation, and test sets
        train_size = int(0.8 * len(filtered_image_paths))
        val_size = int(0.2 * len(filtered_image_paths))

        train_set.extend([(img_path, class_idx, class_name) for img_path in filtered_image_paths[:train_size]])
        val_set.extend([(img_path, class_idx, class_name) for img_path in filtered_image_paths[train_size:train_size+val_size]])

    # After dataset split, print the number of images in the training set for each class
    class_counts = {class_name: 0 for class_name in class_names}
    for _, _, class_name in train_set:
        class_counts[class_name] += 1

    # Now print the number of images used in the training set for each class
    print("\nNumber of images used for training in each class:")
    for class_name, count in class_counts.items():
        print(f"Class: {class_name}, Training images: {count}")

    # Write corrupted image paths to a file
    corrupted_images_file = os.path.join(output_dir, 'corrupted_images.txt')
    with open(corrupted_images_file, 'w') as f:
        f.write('\n'.join(corrupted_images))

    # Shuffle train, validation, and test sets
    random.shuffle(train_set)
    random.shuffle(val_set)

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
                image = Image.open(img_path).convert("L")
            else:
                image = img_path
            if self.transform:
                image = self.transform(image)
            return image, label, class_name

    # Create custom datasets
    train_dataset = CustomDataset(train_set, transform=transform)
    val_dataset = CustomDataset(val_set, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load pre-trained model
    resnet_model = timm.create_model('resnet50', pretrained=True, drop_rate=0.3, num_classes=len(class_names))

    # Move model to device
    resnet_model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=learning_rate, weight_decay=1e-3)
    patience = 10

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)

    # Log file
    log_file_path = os.path.join(output_dir, 'training_log.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch\tTrain Loss\tVal Loss\tVal Accuracy\n")

        # Define the number of epochs, test interval, and patience for early stopping
        num_epochs = 7777777

        # Initialize lists to store metrics at each step
        train_step_losses = []
        val_step_losses = []
        val_step_accuracies = []

        # Initialize variables for early stopping
        best_val_accuracy = 0.0
        epochs_since_improvement = 0

        for epoch in range(num_epochs):
            resnet_model.train()
            running_loss = 0.0
            for batch_idx, (inputs, labels, class_names) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = resnet_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                train_step_losses.append(loss.item())  # Record train loss at each step

                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            # Training epoch finished
            epoch_loss = running_loss / len(train_set)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')

            # Update the learning rate scheduler after each epoch
            scheduler.step()

            # Validation at the end of each epoch
            resnet_model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels, _ in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = resnet_model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(val_set)
            val_accuracy = correct / total

            val_step_losses.append(val_loss)  # Record validation loss at each step
            val_step_accuracies.append(val_accuracy)  # Record validation accuracy at each step

            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%}')

            # Check for early stopping
            if val_accuracy > best_val_accuracy:

                # Save the trained model
                model_save_path = os.path.join(output_dir, 'model.bin')
                torch.save(resnet_model.state_dict(), model_save_path)
                print(f"Trained model saved to: {model_save_path}")

                best_val_accuracy = val_accuracy
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered. Validation accuracy did not improve within patience epochs.")
                break  # Stop training

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plotting Train Loss on the left y-axis
        ax1.plot(range(len(train_step_losses)), train_step_losses, label='Train Loss', color='tab:blue')
        ax1.plot(range(len(val_step_losses)), val_step_losses, label='Val Loss', color='tab:orange')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Creating a secondary y-axis for Val Accuracy
        ax2 = ax1.twinx()
        ax2.plot(range(len(val_step_accuracies)), val_step_accuracies, label='Val Accuracy', color='tab:green')
        ax2.set_ylabel('Accuracy (%)')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Adding legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center right')

        plt.title('Training and Validation Metrics')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'training_validation_plot.png'), dpi=600)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <train/test image directory> <output directory>")
        sys.exit(1)

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(data_dir, output_dir)

