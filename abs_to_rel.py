import json

import config
import util

# TODO: Set the dir

dataset_dir = config.project_dir+'samson_easy_gt/'

###########################################################################

instances_file_path = dataset_dir+'instances.json'

with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)

for image_id, instance in instances.items():

    for annotation in instance['annotations']:

        util.abs_to_rel(
            annotation['bbox'], instance['width'], instance['height'])

        util.norm_rel_bbox(annotation['bbox'])

util.write_json_file(instances, instances_file_path)
