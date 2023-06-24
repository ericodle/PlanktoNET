from PIL import Image, ImageTk
import os
import csv
import tkinter as tk

# Path to the folder containing the images
folder_path = r"D:\Uni\Dino\Projects\IFCB\png\test"

# List all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith((".png"))]

# Create a CSV file
csv_file = open("image_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Image Name", "taxid"])

# Initialize GUI
root = tk.Tk()
root.title("Image Viewer")
root.geometry("500x500")

# Create labels for image name, width, and height
image_name_label = tk.Label(root, text="Image Name:")
image_name_label.pack()

# Display the images
current_image_index = 0

def display_image():
    global current_image_index

    if current_image_index < len(image_files):
        image_file = image_files[current_image_index]
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        # Update labels with image information
        image_name_label.config(text="Image Name: " + image_file)
        # Create entry field for ID input
        id_label = tk.Label(root, text="Enter ID:")
        id_label.pack()

        id_entry = tk.Entry(root)
        id_entry.pack()
        id_entry.focus()

        # Function to save image data and proceed to the next image
        def save_and_proceed():
            global current_image_index

            # Write image data to CSV file
            image_id = id_entry.get()
            csv_writer.writerow([image_file, image_id])

            # Clear the ID entry field
            id_entry.delete(0, tk.END)

            # Destroy the ID input widgets
            id_label.pack_forget()
            id_entry.pack_forget()
            image_label.pack_forget()
            save_button.pack_forget()
            # Proceed to the next image
            current_image_index += 1
            display_image()

        # Create a button for saving the ID and proceeding to the next image
        save_button = tk.Button(root, text="Save ID and Next", command=save_and_proceed)
        save_button.pack()
        
        # Show the image
        photo = ImageTk.PhotoImage(image)
        if "image_label" in globals():
            image_label.configure(image=photo)
            image_label.image = photo  # Update the reference to avoid garbage collection
        else:
            image_label = tk.Label(root, image=photo)
            image_label.image = photo  # Keep a reference to avoid garbage collection
            image_label.pack()

    else:
        # Close the CSV file
        csv_file.close()
        root.quit()

# Start displaying the images
display_image()

# Start the GUI main loop
root.mainloop()
