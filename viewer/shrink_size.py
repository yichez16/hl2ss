import os
from PIL import Image

def resize_images_in_folder(source_folder, target_folder, target_size=(640, 192)):
    """
    Resize all images in the source_folder and save them to the target_folder with the target_size.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            source_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, file_name)

            with Image.open(source_path) as img:
                img = img.resize(target_size)
                img.save(target_path)

    print(f"All images resized and saved in '{target_folder}'")

# Usage
source_folder = './data/rgb'  # Replace with the path to your source folder
target_folder = './data/rgb_640'  # Replace with the path to your target folder
resize_images_in_folder(source_folder, target_folder)