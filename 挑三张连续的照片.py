import os
import shutil
import re

def find_and_copy_consecutive_images(folders, target_folder, prefix='sp'):
    """
    Find consecutive images in the given folders, rename and copy them to the target folder.

    Parameters:
    - folders: A list of folder paths to search for consecutive images.
    - target_folder: The folder path where the consecutive images will be copied.
    - prefix: The prefix to add to the filenames when copying.
    """
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)

    # Regular expression to match file names like 'ECSP0001.JPG'
    pattern = re.compile(r'(IMG_)(\d+)\.JPG', re.IGNORECASE)

    for folder in folders:
        # Get a list of all JPG files in the folder
        files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
        files.sort()  # Sort files alphabetically by name

        # List to hold file names and their extracted numbers
        file_data = []

        # Extract number parts and store them with the filename
        for file in files:
            match = pattern.match(file)
            if match:
                num_part = int(match.group(2))  # Convert the numeric part to an integer
                file_data.append((file, num_part))

        # Check for consecutive files and copy them
        for i in range(len(file_data) - 2):
            # Check if the next two files are consecutive
            if file_data[i + 1][1] == file_data[i][1] + 1 and file_data[i + 2][1] == file_data[i][1] + 2:
                # Copy and rename the consecutive files to the target folder
                for index, (old_name, _) in enumerate(file_data[i:i+3], start=1):
                    new_name = f"{prefix}{file_data[i][1]}_{index}.jpg"
                    shutil.copy(os.path.join(folder, old_name), os.path.join(target_folder, new_name))
                    print(f"Copied and renamed {old_name} to {target_folder}/{new_name}")

if __name__ == "__main__":
    # List of folders to search for consecutive images
    folders_to_search = [
        r"F:\金寨县\天堂寨\TTZ-013",
        # Add more folders as needed
    ]

    # Target folder where consecutive images will be copied
    target_folder_path = r"D:\光流照片"

    # Call the function
    find_and_copy_consecutive_images(folders_to_search, target_folder_path)
