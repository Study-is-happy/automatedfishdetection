import json

import config
import util

instances_file_path = config.project_dir+'update/instances.json'

with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)

for _, instance in instances.items():

    annotations = instance['annotations']

    for index, annotation in enumerate(annotations):

        if 'overlap' not in annotation:
            for unchecked_annotation in annotations[index+1:]:
                if util.get_bboxes_iou(annotation['bbox'], unchecked_annotation['bbox']) > 0.5:
                    unchecked_annotation['overlap'] = True

    instance['annotations'] = [
        annotation for annotation in annotations if 'overlap' not in annotation]

util.write_json_file(instances, instances_file_path)
