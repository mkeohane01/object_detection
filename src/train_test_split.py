import os
import shutil
import random

def split_dataset(images_dir, labels_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    # Create directories for the split datasets if they don't already exist
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    random.shuffle(image_files)  # Randomly shuffle the list
    
    # Calculate split indices
    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Function to move files
    def move_files(files, split):
        for f in files:
            # Move image
            shutil.move(os.path.join(images_dir, f), os.path.join(images_dir, split, f))
            # Move corresponding label file
            label_file = os.path.splitext(f)[0] + '.txt'
            shutil.move(os.path.join(labels_dir, label_file), os.path.join(labels_dir, split, label_file))
    
    # Split and move files
    move_files(image_files[:train_end], 'train')
    move_files(image_files[train_end:val_end], 'val')
    move_files(image_files[val_end:], 'test')

if __name__ == "__main__":
    images_dir = 'data/yolo-data/images'
    labels_dir = 'data/yolo-data/labels'

    # Execute the function
    split_dataset(images_dir, labels_dir)
