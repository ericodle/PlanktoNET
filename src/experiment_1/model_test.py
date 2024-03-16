import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt

def evaluate_sorting_performance(model, test_loader, device):
    predicted_labels = []
    actual_labels = []

    # Predict labels for test images
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())

    # Sort images based on predicted labels
    sorted_indices = sorted(range(len(predicted_labels)), key=lambda i: predicted_labels[i])
    sorted_images = [test_loader.dataset.samples[i][0] for i in sorted_indices]
    sorted_actual_labels = [actual_labels[i] for i in sorted_indices]

    # Evaluate sorting performance (accuracy)
    accuracy = accuracy_score(actual_labels, predicted_labels)

    return sorted_images, sorted_actual_labels, accuracy

def save_sorted_images(images, labels, output_dir, input_dataset):
    os.makedirs(output_dir, exist_ok=True)
    for i, image_path in enumerate(images):
        img = Image.open(image_path)
        label = input_dataset.classes[labels[i]]
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        img_name = f"sorted_image_{i}.png"
        img.save(os.path.join(label_dir, img_name))

def main(data_dir, model_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(dataset.classes))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    sorted_images, sorted_actual_labels, accuracy = evaluate_sorting_performance(model, test_loader, device)

    print(f"Accuracy of sorting: {accuracy}")

    # Save sorted images to the output directory
    save_sorted_images(sorted_images, sorted_actual_labels, output_dir, dataset)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <test image directory> <path to pre-trained model> <output directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]
    main(data_dir, model_path, output_dir)
