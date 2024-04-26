import os
import sys
import shutil
from PIL import Image

def is_corrupted(file_path):
    try:
        with open(file_path, 'rb') as f:
            f.read(1)
        return False
    except Exception:
        return True

def cut_bottom_pixels(image_path, pixels):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # Check if the height is at least three times the amount being cut
            if height >= 10 * pixels:
                img = img.crop((0, 0, width, height - pixels))
                img.save(image_path)
            else:
                print(f"Skipping image {image_path} due to insufficient height for cropping.")
    except Exception as e:
        print(f"Error while processing image {image_path}: {e}")

def generate_files(input_directory, output_directory, cut_bottom):
    # List to store names of subdirectories
    subdirectories = []

    # Number of pixels to cut from the bottom
    cut_pixels = 12  # Adjust this value as needed

    # Iterate through each file in the input directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            input_file_path = os.path.join(root, file)
            output_file_path = os.path.join(output_directory, root[len(input_directory)+1:], file)
            if is_corrupted(input_file_path):
                print(f"Removing corrupted file: {input_file_path}")
                continue
            try:
                if cut_bottom:
                    cut_bottom_pixels(input_file_path, cut_pixels)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                shutil.copy(input_file_path, output_file_path)  # Copy the file to the output directory
            except Exception as e:
                print(f"Error while processing file {input_file_path}: {e}")
                continue  # Skip copying the file if an error occurs

        # Add subdirectory name to the list if it's not the current directory
        if root != input_directory:
            subdirectories.append(os.path.relpath(root, input_directory))

    # Write the subdirectory names to new_classes.txt
    #with open(os.path.join(output_directory, 'new_classes.txt'), 'w') as f:
        #for subdir in subdirectories:
            #f.write(subdir + '\n')

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input directory> <output directory>")
        sys.exit(1)
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    generate_files(input_directory, output_directory, cut_bottom=True)

if __name__ == "__main__":
    main()
