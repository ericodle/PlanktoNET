import os
import random
import shutil
import sys

'''
This script processes a root folder containing subfolders of images, ensuring each subfolder retains only 100 randomly selected images by removing any excess. It validates inputs, iterates through subfolders, and applies selection logic using Python's os and random modules.
'''

# Select and keep only 100 random images in each subfolder
def select_and_keep_images(folder_path):
    for subdir, _, files in os.walk(folder_path):
        if len(files) > 100:
            files_to_keep = random.sample(files, 100)
            for file_name in files:
                if file_name not in files_to_keep:
                    os.remove(os.path.join(subdir, file_name))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <root_folder>")
        sys.exit(1)

    root_folder = sys.argv[1]

    # Check if the provided path is a directory
    if not os.path.isdir(root_folder):
        print("Error: The provided path is not a directory.")
        sys.exit(1)

    # Iterate through subfolders and select images to keep
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            select_and_keep_images(folder_path)
