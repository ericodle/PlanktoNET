import os
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def extract_features(data_dir, max_images_per_class=30):
    features = []
    labels = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for class_idx, class_name in enumerate(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        image_paths = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path)]
        if len(image_paths) > max_images_per_class:
            image_paths = np.random.choice(image_paths, max_images_per_class, replace=False)

        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).numpy().flatten()  # Flatten the image to a 1D array
            features.append(image_tensor)
            labels.append(class_idx)

    features = np.array(features)
    labels = np.array(labels)
    return features, labels

def main(data_dir, output_dir):
    # Step 1: Extract features from images
    features, true_labels = extract_features(data_dir)

    # Step 2: Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Step 3: Get the number of clusters (equal to the number of unique classes)
    num_clusters = len(np.unique(true_labels))

    # Step 4: Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(features_normalized)

    # Step 5: Create clusters dictionary
    clusters = defaultdict(list)
    for img_idx, cluster_label in enumerate(cluster_labels):
        clusters[str(cluster_label)] = []  # Convert the key to string
    for img_idx, cluster_label in enumerate(cluster_labels):
        clusters[str(cluster_label)].append(img_idx)  # Convert the key to string

    # Step 6: Evaluate clustering accuracy
    predicted_labels = np.zeros_like(true_labels)
    for cluster_label, cluster_images in clusters.items():
        predicted_labels[cluster_images] = int(cluster_label)  # Convert the key back to int for indexing

    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"Clustering Accuracy: {accuracy:.2f}")

    # Step 7: Save cluster information and accuracy to a JSON file in the output directory
    output_file = os.path.join(output_dir, 'clusters.json')
    with open(output_file, 'w') as f:
        json.dump({"clusters": clusters, "accuracy": accuracy}, f, indent=4)

    print(f"Cluster information and accuracy saved to {output_file}")

    # Step 8: Plot the clustering results using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    features_reduced = pca.fit_transform(features_normalized)

    plt.figure(figsize=(10, 8))
    for i in range(num_clusters):
        plt.scatter(features_reduced[cluster_labels == i, 0], features_reduced[cluster_labels == i, 1], label=f'Cluster {i}')

    plt.title('Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'clusters_plot.png'))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <data_directory> <output_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(data_dir, output_dir)

