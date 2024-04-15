from PIL import Image
import os
import sys
import shutil

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
            img = img.crop((0, 0, width, height - pixels))
            img.save(image_path)
    except Exception as e:
        print(f"Error while processing image {image_path}: {e}")

def generate_files(input_directory, output_directory, cut_bottom):
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
                    cut_bottom_pixels(input_file_path, 12)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                shutil.copy(input_file_path, output_file_path)  # Copy the file to the output directory
            except Exception as e:
                print(f"Error while processing file {input_file_path}: {e}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <input directory> <output directory> <cut bottom pixels (true/false)>")
        sys.exit(1)
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    cut_bottom = sys.argv[3].lower() == 'true'

    generate_files(input_directory, output_directory, cut_bottom)

if __name__ == "__main__":
    main()
