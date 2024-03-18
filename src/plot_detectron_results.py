import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import json

# class to label mapping
class_mapping = {
    0: 'drink',
    1: 'laptop',
    2: 'utensil',
}
# Assume `data` is your JSON-loaded object
data = json.load(open('data/results/detectron_results_2.json'))  # Adjust the path as needed

# Group annotations by image_id
annotations_by_image = {}
for item in data:
    image_id = item['image_id']
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(item)

# Select 16 random images
selected_images = random.sample(list(annotations_by_image.keys()), 24)

fig, axes = plt.subplots(4, 6, figsize=(25, 25))  # Adjust the size as needed
axes = axes.flatten()

for ax, image_id in zip(axes, selected_images):
    image_path = f'data/yolo-data/images/test/{image_id}'  # Adjust the path as needed
    img = Image.open(image_path)
    ax.imshow(img)
    ax.axis('off')  # Hide axes
    for annotation in annotations_by_image[image_id]:
        if annotation['score'] > 0.5:
            bbox = annotation['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], f"{class_mapping[annotation['category_id']]}: {annotation['score']:.2f}", color='white', fontsize=12)

plt.tight_layout()
plt.show()
