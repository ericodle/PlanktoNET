import os
import sys
import shutil
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

'''
This script uses a pretrained ResNet101 model to classify images from a specified directory, organizing them into subfolders within an output directory based on predicted classes. It ensures robust image handling, GPU utilization if available, and generates a confidence distribution plot of predictions.
'''

class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        # Iterate over files in the root directory
        for filename in os.listdir(root):
            filepath = os.path.join(root, filename)
            # Attempt to open the image, skip if it fails
            try:
                Image.open(filepath).verify()
                self.samples.append(filepath)
            except (IOError, SyntaxError) as e:
                print(f"Skipping {filepath}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = Image.open(path).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

def sort_images(model, dataset, output_dir, class_names):
    device = next(model.parameters()).device
    model.eval()

    confidences = []

    with torch.no_grad():
        for inputs, paths in dataset:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidences_batch, predicted = torch.max(probabilities, 1)

            # Append batch confidences to the list
            confidences.extend(confidences_batch.cpu().numpy())  # Move to CPU

            for idx, pred in enumerate(predicted):
                class_name = class_names[pred.item()]
                confidence = confidences_batch[idx].item()
                image_path = paths[idx]
                image_name = os.path.basename(image_path)
                image_name_with_confidence = f"{os.path.splitext(image_name)[0]}_{confidence:.4f}{os.path.splitext(image_name)[1]}"
                output_subfolder = os.path.join(output_dir, class_name)
                os.makedirs(output_subfolder, exist_ok=True)
                output_path = os.path.join(output_subfolder, image_name_with_confidence)
                shutil.copy(image_path, output_path)
                print(f"Copied {image_name} to {class_name} with confidence {confidence:.4f}")

    # Plot and save confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()

def load_resnet_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.txt')
    with open(class_names_path, 'r') as f:
        class_names = f.read().splitlines()

    num_classes = len(class_names)

    resnet101_model = torchvision.models.resnet101(pretrained=False)
    num_ftrs = resnet101_model.fc.in_features
    resnet101_model.fc = torch.nn.Linear(num_ftrs, num_classes)
    resnet101_model.load_state_dict(torch.load(model_path))

    return resnet101_model, class_names

def main(data_dir, model_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Load model and class names
    model, class_names = load_resnet_model(model_path)
    model.to(device)

    # Load dataset and create data loader
    dataset = CustomImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Sort images into subfolders based on model predictions
    sort_images(model, data_loader, output_dir, class_names)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <image directory> <model checkpoint> <output directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]
    main(data_dir, model_path, output_dir)
