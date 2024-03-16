import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt

def main(data_dir, output_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes

    print(f"Number of classes found in the dataset: {len(class_names)}")
    print("Classes found in the dataset:")
    for cls in class_names:
        print(cls)

    test_set = dataset

    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(class_names))
    vit_model.load_state_dict(torch.load(model_path))
    print("Trained model parameters loaded.")
    vit_model.to(device)
    vit_model.eval()

    predictions = []
    ground_truths = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            print("Outputs being predicted from inputs.")
            outputs = vit_model(inputs)
            _, predicted = torch.max(outputs, 1)
            print("t1.")
            predictions.extend(predicted.cpu().numpy())
            print("t2.")
            ground_truths.extend(labels.numpy())
            print("t3.")

    predictions = np.array(predictions)
    print("t4.")
    ground_truths = np.array(ground_truths)
    print("t5.")
    confusion = confusion_matrix(ground_truths, predictions)
    print("t6.")
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, average='weighted')
    recall = recall_score(ground_truths, predictions, average='weighted')
    f1 = f1_score(ground_truths, predictions, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <test image directory> <output directory> <path to pre-trained model>")
        sys.exit(1)
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3]
    main(data_dir, output_dir, model_path)
