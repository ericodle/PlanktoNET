import os
import random
import sys
import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

'''
This script fine-tunes a ResNet101 model on a custom dataset by loading pretrained weights, handling data augmentation and corrupted images, training the model, and logging and visualizing training metrics, while also implementing early stopping based on test accuracy.
'''

class_names_file = "class_names.txt"  # Define the file name for class names

def load_custom_resnet101(weights_path):
    """
    Load a custom ResNet-101 model from a local file.
    """
    # Load class names from pretrained class_names.txt
    class_names_path = os.path.join(os.path.dirname(weights_path), "class_names.txt")
    with open(class_names_path, 'r') as f:
        original_class_names = [line.strip() for line in f.readlines()]

    # Create an instance of the ResNet-101 model
    model = models.resnet101(weights=ResNet101_Weights.DEFAULT)

    # Modify the last fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(original_class_names))

    # Load the weights from the specified file
    model.load_state_dict(torch.load(weights_path))

    return model, original_class_names

def main(data_dir, output_dir, model_path, learning_rate, num_imgs):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define transforms for data augmentation and normalization
    transform_list = [
        transforms.Resize((256, 256)),  # Resize images to a fixed size
        transforms.CenterCrop(224),  # Crop the center of the image to 224x224
        transforms.ToTensor()
    ]

    transform = transforms.Compose(transform_list)

    # Load dataset and filter out classes with no images
    dataset = datasets.ImageFolder(root=data_dir, transform=None)
    new_class_names = [class_name for class_name in dataset.classes if len(os.listdir(os.path.join(data_dir, class_name))) > 0]

    # Load the custom ResNet-101 model and get original class names
    resnet101_model, original_class_names = load_custom_resnet101(model_path)

    # Modify the final fully-connected layer
    num_original_classes = len(original_class_names)
    num_new_classes = len(new_class_names)
    
    num_ftrs = resnet101_model.fc.in_features
    resnet101_model.fc = nn.Linear(num_ftrs, num_original_classes + num_new_classes)

    print(resnet101_model)

    # Log and store class names
    with open(os.path.join(output_dir, class_names_file), 'w') as f:
        f.write('\n'.join(original_class_names))
        f.write('\n'.join(new_class_names))

    # Printout of classes
    print(f"Number of classes found in pretrained model: {len(original_class_names)}")
    for cls in original_class_names:
        print(cls)

    print(f"Number of classes found in finetune dataset: {len(new_class_names)}")
    print("Classes:")
    for cls in new_class_names:
        print(cls)

    # Function to check if an image file is corrupted
    def is_corrupted_image(file_path):
        try:
            Image.open(file_path).verify()
            return False
        except (IOError, SyntaxError):
            return True

    # Split dataset into train and test sets
    train_set = []
    test_set = []
    corrupted_images = []

    for class_idx, class_name in enumerate(new_class_names):
        class_path = os.path.join(data_dir, class_name)
        image_paths = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path)]

        # Pre-screen training images for corruption
        filtered_image_paths = []
        for img_path in image_paths:
            if not is_corrupted_image(img_path):
                filtered_image_paths.append(img_path)
            else:
                corrupted_images.append(img_path)

        if len(filtered_image_paths) > num_imgs:
            filtered_image_paths = random.sample(filtered_image_paths, num_imgs)  # Randomly select desired number of images

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
        train_size = int(0.9 * len(filtered_image_paths))
        train_set.extend([(img_path, class_idx, class_name) for img_path in filtered_image_paths[:train_size]])
        test_set.extend([(img_path, class_idx, class_name) for img_path in filtered_image_paths[train_size:]])

    # Write corrupted image paths to a file
    corrupted_images_file = os.path.join(output_dir, 'corrupted_images.txt')
    with open(corrupted_images_file, 'w') as f:
        f.write('\n'.join(corrupted_images))

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
    
    # Move model to device
    resnet101_model = resnet101_model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet101_model.parameters(), lr=learning_rate)

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)

    # Log file
    log_file_path = os.path.join(output_dir, 'training_log.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch\tTrain Loss\tTest Loss\tTest Accuracy\n")

        # Define the number of epochs, test interval, and patience for early stopping
        num_epochs = 100
        test_interval = 100  # Test every 100 steps

        # Initialize lists to store metrics at each step
        train_step_losses = []
        test_step_losses = []
        test_step_accuracies = []

        # Initialize variables for early stopping
        best_test_accuracy = 0.0
        epochs_since_improvement = 0

        for epoch in range(num_epochs):
            resnet101_model.train()
            running_loss = 0.0
            for batch_idx, (inputs, labels, new_class_names) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = resnet101_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                train_step_losses.append(loss.item())  # Record train loss at each step

                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

                # Save transformed images
                for i in range(inputs.size(0)):
                    class_output_dir = os.path.join(output_dir, new_class_names[i])
                    if not os.path.exists(class_output_dir):
                        os.makedirs(class_output_dir)
                    output_path = os.path.join(class_output_dir, f'image_{epoch}_{batch_idx}_{i}.jpg')
                    torchvision.utils.save_image(inputs[i], output_path)

                # Test every few steps
                if batch_idx % test_interval == 0:
                    resnet101_model.eval()
                    test_loss = 0.0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, labels, _ in test_loader:
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            outputs = resnet101_model(inputs)
                            loss = criterion(outputs, labels)
                            test_loss += loss.item() * inputs.size(0)
                            _, predicted = torch.max(outputs, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    test_loss /= len(test_set)
                    test_accuracy = correct / total

                    test_step_losses.append(test_loss)  # Record test loss at each step
                    test_step_accuracies.append(test_accuracy)  # Record test accuracy at each step

                    resnet101_model.train()  # Switch back to training mode

            # Training epoch finished
            epoch_loss = running_loss / len(train_set)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')

            # Update the learning rate scheduler after each epoch
            scheduler.step()

            # Testing at the end of each epoch
            resnet101_model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = resnet101_model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            test_loss /= len(test_set)
            test_accuracy = correct / total

            test_step_losses.append(test_loss)  # Record test loss at each step
            test_step_accuracies.append(test_accuracy)  # Record test accuracy at each step

            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}')

            # Check for early stopping
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= 1:
                print("Early stopping triggered. Test accuracy did not improve within patience epochs.")
                break  # Stop training

        # Calculate the number of steps per epoch
        steps_per_epoch = len(train_loader)

        # Calculate the number of epochs completed
        num_completed_epochs = min(epoch + 1, num_epochs)

        # Create a list to store the epoch numbers
        epochs = [i for i in range(num_completed_epochs)]

        # Calculate the x-coordinates of epoch marks
        epoch_marks = [i * steps_per_epoch for i in epochs]

        # Plotting
        num_train_steps = len(train_step_losses)
        num_test_steps = len(test_step_losses)

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plotting Train Loss on the left y-axis
        ax1.plot(range(num_train_steps), train_step_losses, label='Train Loss', color='tab:blue')
        ax1.plot(np.linspace(0, num_train_steps-1, num_test_steps), test_step_losses, label='Test Loss', color='tab:orange')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Creating a secondary y-axis for Test Accuracy
        ax2 = ax1.twinx()
        ax2.plot(np.linspace(0, num_train_steps-1, num_test_steps), test_step_accuracies, label='Test Accuracy', color='tab:green')
        ax2.set_ylabel('Accuracy (%)')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Adding x-axis marks for epochs
        ax1.set_xticks(epoch_marks)
        ax1.set_xticklabels(epochs)

        # Adding legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center right')

        plt.title('Training and Testing Metrics at Each Step')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'training_plot_steps_with_epochs.png'), dpi=600)


    # Save the trained model
    model_save_path = os.path.join(output_dir, 'model.bin')
    torch.save(resnet101_model.state_dict(), model_save_path)
    print(f"Trained model saved to: {model_save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python script.py <train/test image directory> <output directory> <initial learning rate> <images per class>")
        sys.exit(1)
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3]
    learning_rate = float(sys.argv[4])
    num_imgs = int(sys.argv[5])
    main(data_dir, output_dir, model_path, learning_rate, num_imgs)
