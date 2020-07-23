import os
import json
import numpy as np

import config
import util

# TODO: Set the dirs

dataset_dir = config.project_dir+'fake_gt/'

###########################################################################

instances_file_path = dataset_dir+'instances.json'

with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)

for image_id, instance in instances.items():

    image_width = instance['width']
    image_height = instance['height']

    fish_annotations = []
    sponge_annotations = []
    annotations = []

    for annotation in instance['annotations']:

        if annotation['category_id'] == 0:
            fish_annotations.append(annotation)

        else:
            annotations.append(annotation)

            if annotation['category_id'] == 2:
                sponge_annotations.append(annotation)

    for fish_annotation in fish_annotations:
        for sponge_annotation in sponge_annotations:
            if util.get_bbox1_intersection(fish_annotation['bbox'], sponge_annotation['bbox']) > 0.5:
                break
        else:
            annotations.append(fish_annotation)

    instance['annotations'] = annotations

util.write_json_file(instances, instances_file_path)
