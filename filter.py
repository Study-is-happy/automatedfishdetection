import os
import json
import numpy as np

import config
import utils

# TODO: Set the dirs

dataset_dir = config.project_dir + 'update/'

###########################################################################

instances_file_path = dataset_dir + 'instances_1.json'

with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)

    for image_id, instance in instances.items():

        image_width = instance['width']
        image_height = instance['height']

        annotations = []

        for annotation in instance['annotations']:

            if annotation['category_id'] == 0:
                annotation['category_id'] = 1
                annotations.append(annotation)
            elif annotation['category_id'] == 2:
                annotation['category_id'] = 9
                annotations.append(annotation)

        instance['annotations'] = annotations

utils.write_json_file(instances, instances_file_path)
