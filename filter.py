import os
import json
import numpy as np

import config
import util

# TODO: Set the dirs

dataset_dir = config.project_dir+'update/'

###########################################################################

instances_file_path = dataset_dir+'instances.json'

with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)

    for image_id, instance in instances.items():

            image_width = instance['width']
            image_height = instance['height']

            annotations = []

            for annotation in instance['annotations']:

                if annotation['category_id'] != len(config.categories)-1:
                    annotations.append(annotation)

            instance['annotations'] = annotations

util.write_json_file(instances, instances_file_path)
