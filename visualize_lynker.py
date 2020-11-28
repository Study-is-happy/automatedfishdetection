import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

import config

# TODO: Set the dirs

dataset_dir = config.project_dir+'lynker/'

###########################################################################

images_dir = dataset_dir+'images/'

instances_file_path = dataset_dir+'instances.json'

with open(instances_file_path) as instances_file:

    instances_dict = json.load(instances_file)

for image_id in sorted(instances_dict):

    print(image_id)

    instance = instances_dict[image_id]

    # if image_id != '20100922.163718.01228':
    #     continue

    # if len(instance['annotations']) < 20:
    #     continue

    instance = instances_dict[image_id]

    image = mpimg.imread(images_dir+image_id+'.jpg')

    width = instance['width']
    height = instance['height']

    fig = plt.figure(figsize=(20, 20))

    plt.title(image_id)

    plt.imshow(image)

    current_axis = plt.gca()

    for annotation in instance['annotations']:

        bbox = annotation['bbox']

        current_axis.add_patch(plt.Rectangle(
            (bbox[0]*width, bbox[1]*height), (bbox[2]-bbox[0])*width, (bbox[3]-bbox[1])*height, fill=False, linewidth=3))

        plt.text(bbox[0]*width, bbox[1]*height-3,
                 config.lynker_categories[annotation['category_id']], size=30, bbox={'alpha': 0.5, 'pad': 3})

    plt.show()