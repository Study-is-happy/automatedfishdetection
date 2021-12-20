import os
import json
import numpy as np

import config
import utils

# TODO: Set the dirs

dataset_dir = config.project_dir + 'all/'

###########################################################################

category_instances = {}

with open(dataset_dir + 'instances_all.json') as instances_file:
    instances = json.load(instances_file)

    for image_id, instance in instances.items():

        image_width = instance['width']
        image_height = instance['height']

        category_annotations = []
        background_annotations = []

        for annotation in instance['annotations']:

            if annotation['category_id'] == 1:
                annotation['category_id'] = 0
                category_annotations.append(annotation)
            elif annotation['category_id'] == 4 or annotation['category_id'] == 5 or annotation['category_id'] == 6:
                annotation['category_id'] = 1
                background_annotations.append(annotation)

        len_category_annotations = len(category_annotations)
        len_background_annotations = len(background_annotations)

        if len_category_annotations > 0:

            # count_threshold = len(category_annotations) * 3

            # if len(background_annotations) > count_threshold:
            #     background_annotations = background_annotations[:count_threshold]

            if len_category_annotations < 100:

                if len_category_annotations + len_background_annotations > 100:

                    background_annotations = background_annotations[:min(100 - len_category_annotations, len_background_annotations)]

                    # print(len(category_annotations) + len(background_annotations))

                category_instances[image_id] = {'width': image_width, 'height': image_height, 'annotations': category_annotations + background_annotations}

utils.write_json_file(category_instances, dataset_dir + 'instances.json')
