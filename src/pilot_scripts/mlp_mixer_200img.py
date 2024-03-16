import os
import random
import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, input_dim)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.token_mixing = MlpBlock(tokens_mlp_dim, tokens_mlp_dim)
        self.channel_mixing = MlpBlock(tokens_mlp_dim, channels_mlp_dim)
        self.layer_norm1 = nn.LayerNorm(tokens_mlp_dim)
        self.layer_norm2 = nn.LayerNorm(tokens_mlp_dim)
    
    def forward(self, x):
        y = self.layer_norm1(x)
        y = y.permute(0, 1, 2)
        y = self.token_mixing(y)
        y = y.permute(0, 1, 2)
        x = x + y
        y = self.layer_norm2(x)
        y = self.channel_mixing(y)
        return x + y

class MlpMixer(nn.Module):
    def __init__(self, patches, num_classes, num_blocks, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MlpMixer, self).__init__()
        self.patches = patches
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        
        self.stem = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patches, stride=patches)
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.pre_head_layer_norm = nn.LayerNorm(hidden_dim)
        if num_classes:
            self.head = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.pre_head_layer_norm(x)
        x = x.mean(dim=1)
        if self.num_classes:
            x = self.head(x)
        return x

def main(data_dir, output_dir, learning_rate):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = ImageFolder(root=data_dir, transform=None)
    num_classes = len(dataset.classes)

    # Split dataset into train and test sets
    train_set = []
    test_set = []
    for class_idx, class_name in enumerate(dataset.classes):
        class_path = os.path.join(data_dir, class_name)
        image_paths = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path)]
        if len(image_paths) > 200:
            image_paths = random.sample(image_paths, 200)  # Randomly select 200 images
        
        # Augment existing images to create additional training samples
        if len(image_paths) < 200:
            augmentation_transform = transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])
            num_additional_samples = 200 - len(image_paths)
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

    # Load MlpMixer model
    model = MlpMixer(patches=224, num_classes=num_classes, num_blocks=224, hidden_dim=224, tokens_mlp_dim=224, channels_mlp_dim=224)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

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
            model.train()
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
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
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
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
    learning_rate = float(sys.argv[3]) 
    main(data_dir, output_dir, learning_rate)

