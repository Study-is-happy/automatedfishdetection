import os
import json
import numpy as np

import config
import utils

# TODO: Set the dirs

dataset_dir = config.project_dir + 'train/'

###########################################################################

category_instances = {}

with open(dataset_dir + 'instances (copy).json') as instances_file:
    instances = json.load(instances_file)

    for image_id, instance in instances.items():

        image_width = instance['width']
        image_height = instance['height']

        annotations = []
        count = 0

        for annotation in instance['annotations']:

            annotations.append(annotation)
            count += 1

        if count < 100:

            category_instances[image_id] = {'width': image_width, 'height': image_height, 'annotations': annotations}

utils.write_json_file(category_instances, dataset_dir + 'instances.json')
