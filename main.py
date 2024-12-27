import sys
import subprocess
import os

def welcome_message():
    print("Welcome to the Training and Evaluation System!")
    print("Please select an option:")

def train_option():
    print("\nYou selected 'Train'!")
    
    # Prompt for training data directory
    train_data_dir = input("Enter the path to the training image directory: ")
    # Check if directory exists
    if not os.path.isdir(train_data_dir):
        print(f"Error: The directory '{train_data_dir}' does not exist.")
        return
    
    # Prompt for output directory
    output_dir = input("Enter the path to the output directory: ")
    # Check if directory exists, or create it
    if not os.path.isdir(output_dir):
        print(f"Warning: The directory '{output_dir}' does not exist. It will be created.")
        os.makedirs(output_dir)
    
    # Run train.py with the provided directories
    try:
        subprocess.run(['python', 'train.py', train_data_dir, output_dir], check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {str(e)}")

def fine_tune_option():
    print("\nYou selected 'Fine-Tune'!")
    # Add logic for fine-tuning (not yet implemented here)
    print("Fine-tuning option is under construction.")

def inference_option():
    print("\nYou selected 'Inference'!")
    # Add logic for inference (not yet implemented here)
    print("Inference option is under construction.")

def evaluate_option():
    print("\nYou selected 'Evaluate'!")
    
    # Prompt for test image directory
    eval_data_dir = input("Enter the path to the test image directory: ")
    # Check if directory exists
    if not os.path.isdir(eval_data_dir):
        print(f"Error: The directory '{eval_data_dir}' does not exist.")
        return
    
    # Prompt for pre-trained model path
    model_path = input("Enter the path to the pre-trained model: ")
    # Check if model file exists
    if not os.path.isfile(model_path):
        print(f"Error: The file '{model_path}' does not exist.")
        return
    
    # Prompt for output directory
    output_dir = input("Enter the path to the output directory: ")
    # Check if directory exists, or create it
    if not os.path.isdir(output_dir):
        print(f"Warning: The directory '{output_dir}' does not exist. It will be created.")
        os.makedirs(output_dir)
    
    # Run eval.py with the provided directories and model path
    try:
        subprocess.run(['python', 'eval.py', eval_data_dir, model_path, output_dir], check=True)
        print("\nEvaluation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {str(e)}")

def main():
    while True:  # This loop keeps the program running after a task is completed
        welcome_message()

        print("\n1. Train")
        print("2. Fine-tune")
        print("3. Inference")
        print("4. Evaluate")
        print("5. Exit")

        choice = input("\nPlease enter the number of your choice (1-5): ")

        if choice == '1':
            train_option()
        elif choice == '2':
            fine_tune_option()
        elif choice == '3':
            inference_option()
        elif choice == '4':
            evaluate_option()
        elif choice == '5':
            print("Exiting the program. Goodbye!")
            break  # Exit the loop and terminate the program
        else:
            print("Invalid option. Please select a number between 1 and 5.")

if __name__ == "__main__":
    main()

