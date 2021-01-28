import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

import config

# TODO: Set the dirs

dataset_dir = config.project_dir + 'train/'

###########################################################################

images_dir = dataset_dir + 'crop_images/'

instances_file_path = dataset_dir + 'crop_instances.json'

with open(instances_file_path) as instances_file:

    instances_dict = json.load(instances_file)

interval = 100
index = -1

for image_id in sorted(instances_dict):

    print(image_id)
    index += 1

    # if index % interval != 0:
    #     continue

    # if image_id != '20171014.194218.01099_rect_color':
    #     continue

    instance = instances_dict[image_id]

    # if len(instance['annotations']) < 10:
    #     continue

    image = mpimg.imread(images_dir + image_id + '.jpg')

    width = instance['width']
    height = instance['height']

    fig = plt.figure(figsize=(20, 20))

    plt.title(image_id)

    plt.imshow(image)

    current_axis = plt.gca()

    exist_category_count = 0

    for annotation in instance['annotations']:

        if annotation['category_id'] == 0:
            exist_category_count += 1

        bbox = annotation['bbox']

        color = config.colors[annotation['category_id']]

        current_axis.add_patch(plt.Rectangle(
            (bbox[0] * width, bbox[1] * height), (bbox[2] - bbox[0]) * width, (bbox[3] - bbox[1]) * height, color=color, fill=False, linewidth=3))

        plt.text(bbox[0] * width, bbox[1] * height - 3,
                 config.categories[annotation['category_id']], color='white', size=10, bbox={'facecolor': color, 'alpha': 0.5, 'pad': 3})

    if exist_category_count >= 0:
        plt.show()
    else:
        plt.close()
