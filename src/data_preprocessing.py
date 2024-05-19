import os
import shutil
from sklearn.model_selection import train_test_split

def create_dataset_directories(base_dir, categories):
    for category in categories:
        os.makedirs(os.path.join(base_dir, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'val', category), exist_ok=True)

def split_dataset(source_dir, target_dir, categories, val_split=0.2):
    for category in categories:
        category_path = os.path.join(source_dir, category)
        images = os.listdir(category_path)
        train_images, val_images = train_test_split(images, test_size=val_split, random_state=42)

        for image in train_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(target_dir, 'train', category, image))

        for image in val_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(target_dir, 'val', category, image))

if __name__ == "__main__":
    source_directory = 'data/raw'
    target_directory = 'data/processed'
    categories = ['benign', 'malignant', 'normal']

    create_dataset_directories(target_directory, categories)
    split_dataset(source_directory, target_directory, categories)
