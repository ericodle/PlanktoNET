import os
import sys

def keep_files(directory):
    # Iterate through each subdirectory
    for root, dirs, files in os.walk(directory):
        # Keep track of the number of files encountered
        file_count = 0
        # Iterate through each file in the subdirectory
        for file in files:
            # Increment the file count
            file_count += 1
            # If there are more than three files, delete the excess files
            if file_count > 20:
                file_path = os.path.join(root, file)
                os.remove(file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <test image directory>")
        sys.exit(1)
    directory_path = sys.argv[1]
    keep_files(directory_path)
