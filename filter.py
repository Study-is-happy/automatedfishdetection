import os
import json

import config
import util

# TODO: Set the dirs

dataset_dirs = [
    config.project_dir+'update/']

###########################################################################

for dataset_dir in dataset_dirs:

    instances_file_path = dataset_dir+'instances.json'

    with open(instances_file_path) as instances_file:
        instances = json.load(instances_file)

    for _, instance in instances.items():

        annotations = []
        for annotation in instance['annotations']:
            if annotation['temp'] == 0:
                annotations.append(annotation)

        instance['annotations'] = annotations

    util.write_json_file(instances, instances_file_path)
