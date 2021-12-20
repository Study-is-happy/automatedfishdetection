import os
import json
import numpy as np

import config
import utils

# TODO: Set the dirs

dataset_dir = config.project_dir + 'fake_gt/'

###########################################################################

instances_file_path = dataset_dir + 'instances.json'

test_count = 0

with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)

    for image_id, instance in instances.items():

        image_width = instance['width']
        image_height = instance['height']

        annotations = []

        for annotation in instance['annotations']:

            fake_fish = False
            if annotation['category_id'] == 0:
                for sponge_annotation in instance['annotations']:
                    if sponge_annotation['category_id'] == 2:
                        if utils.get_bboxes_iou(annotation['bbox'], sponge_annotation['bbox']) > 0.01:
                            fake_fish = True
                            test_count += 1
                            break

            if not fake_fish:
                annotations.append(annotation)

        instance['annotations'] = annotations

utils.write_json_file(instances, instances_file_path)
print(test_count)
