import os
import sys

def generate_folder_tree(folder_path, level=0):
    tree_output = ''
    if level == 0:
        tree_output += os.path.basename(folder_path) + '\n'
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            tree_output += '    ' * (level + 1) + item + '\n'
            tree_output += generate_folder_tree(item_path, level + 1)
    return tree_output

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_tree.py <root_directory_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print("Error: The specified path is not a directory.")
        sys.exit(1)

    folder_tree = generate_folder_tree(folder_path)
    print(folder_tree)

