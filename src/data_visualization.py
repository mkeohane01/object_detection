import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_bounding_boxes(image_path, label_path):
    """Draws bounding boxes on an image given the image and label paths."""
    # Load the image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Image dimensions
    img_width, img_height = image.size
    
    # Load and parse the label file
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            # Assuming YOLO format: class, x_center, y_center, width, height (normalized)
            x_center, y_center, width, height = map(float, parts[1:])
            # Convert to pixel values
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            # Calculate bottom-left corner
            bottom_left_x = x_center - width / 2
            bottom_left_y = y_center - height / 2
            
            # Create a Rectangle patch
            rect = patches.Rectangle((bottom_left_x, bottom_left_y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            
            # add class label
            ax.text(bottom_left_x, bottom_left_y, parts[0], color='r')

            # Add the patch to the Axes
            ax.add_patch(rect)
    
    fig.title = label_path
    plt.show()

def select_random_image_and_draw_boxes(images_dir, labels_dir):
    """Selects a random image and draws its bounding boxes using the corresponding label file."""
    image_files = os.listdir(images_dir)
    if not image_files:
        print("No images found")
        return
    
    selected_image = random.choice(image_files)
    image_path = os.path.join(images_dir, selected_image)
    label_path = os.path.join(labels_dir, os.path.splitext(selected_image)[0] + '.txt')
    
    if not os.path.exists(label_path):
        print(f"No label found for image {selected_image}")
        return
    
    draw_bounding_boxes(image_path, label_path)

if __name__ == "__main__":
    # Paths to the directories
    images_dir = 'data/yolo-data/images/train'
    labels_dir = 'data/yolo-data/labels/train'

    select_random_image_and_draw_boxes(images_dir, labels_dir)
