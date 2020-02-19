import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

import config

# TODO: Set the dirs

dataset_dir = config.project_dir+'update/'

###########################################################################

images_dir = dataset_dir+'images/'

instances_file_path = dataset_dir+'instances.json'

colors = ['red', 'orange', 'white', 'grey']

with open(instances_file_path) as instances_file:

    instances_dict = json.load(instances_file)

for image_id, instance in instances_dict.items():

    print(image_id)

    # if image_id != '20100922.160600.00394':
    #     continue

    if len(instance['annotations']) < 10:
        continue

    image = mpimg.imread(images_dir+image_id+'.jpg')

    width = instance['width']
    height = instance['height']

    fig = plt.figure(figsize=(20, 20))

    plt.title(image_id)

    plt.imshow(image)

    current_axis = plt.gca()

    for annotation in instance['annotations']:

        bbox = annotation['bbox']

        color = colors[annotation['category_id']]

        if 'score' in annotation:
            linestyle = 'dashed'
            plt.text(bbox[0]*width, bbox[1]*height, format(
                annotation['score'], '0.2f'), color=color, size=30)
        else:
            linestyle = '-'

        current_axis.add_patch(plt.Rectangle(
            (bbox[0]*width, bbox[1]*height), (bbox[2]-bbox[0])*width, (bbox[3]-bbox[1])*height, color=color, fill=False, linewidth=3, linestyle=linestyle))

        plt.text(bbox[0]*width, bbox[1]*height-3,
                 config.categories[annotation['category_id']], color='white', size=30, bbox={'facecolor': color, 'alpha': 0.5, 'pad': 3})

    plt.show()
