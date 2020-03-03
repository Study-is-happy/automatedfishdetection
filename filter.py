import os
import json

import config
import util

# TODO: Set the dirs

dataset_dir = config.project_dir+'update/'

###########################################################################

instances_file_path = dataset_dir+'instances.json'

with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)

for image_id, instance in instances.items():

    annotations = []
    for annotation in instance['annotations']:
        # util.norm_bbox(annotation['bbox'])
        # annotations.append(annotation)
        bbox = annotation['bbox']
        if bbox[0] < 0 or bbox[0] > 1 or bbox[1] < 0 or bbox[1] > 1 or bbox[2] < 0 or bbox[2] > 1 or bbox[3] < 0 or bbox[3] > 1:
            print('check')

#     instance['annotations'] = annotations

# util.write_json_file(instances, instances_file_path)
