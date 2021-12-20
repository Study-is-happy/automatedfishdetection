import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

import config

# TODO: Set the dirs

dataset_dir = config.project_dir + 'init/'

###########################################################################

images_dir = dataset_dir + 'images/'

instances_file_path = dataset_dir + 'instances.json'

with open(instances_file_path) as instances_file:

    instances_dict = json.load(instances_file)

index = -1

for image_id in sorted(instances_dict):

    print(image_id)
    index += 1

    # if index % 10 != 0:
    #     continue

    # if image_id != '20100922.160434.00356':
    #     continue

    instance = instances_dict[image_id]

    # if len(instance['annotations']) < 20:
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

        if annotation['category_id'] == 9:
            exist_category_count += 1

        bbox = annotation['bbox']

        color = config.colors[annotation['category_id']]

        if 'score' in annotation:
            text = config.categories[annotation['category_id']] + ': ' + '{:.2f}'.format(annotation['score'])
            linestyle = 'dashed'
        else:
            text = config.categories[annotation['category_id']]
            linestyle = None

        if annotation['category_id'] == 9:
            linestyle = 'dashed'

        current_axis.add_patch(plt.Rectangle(
            (bbox[0] * width, bbox[1] * height), (bbox[2] - bbox[0]) * width, (bbox[3] - bbox[1]) * height, color=color, fill=False, linewidth=3, linestyle=linestyle))
        # current_axis.add_patch(plt.Rectangle(
        #     ((bbox[0] + bbox[2]) / 2 * width - 2, (bbox[1] + bbox[3]) / 2 * height - 2), 4, 4, color=color, fill=False, linewidth=3, linestyle=linestyle))

        plt.text(bbox[0] * width, bbox[1] * height - 3,
                 text, color='white', size=10, bbox={'facecolor': color, 'alpha': 0.5, 'pad': 3})

    if exist_category_count >= 0:
        plt.show()
    else:
        plt.close()
