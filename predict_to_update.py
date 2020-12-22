import os
import json

import config
import util

with open(config.project_dir + 'update/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)

predict_annotations_dir = config.project_dir + 'predict/annotations/'

for annotations_file_name in os.listdir(predict_annotations_dir):

    with open(predict_annotations_dir + annotations_file_name) as predict_annotations_file:
        predict_annotations = json.load(predict_annotations_file)

        for predict_annotation in predict_annotations:
            if 'gt_annotation_index' not in predict_annotation:

                image_id = predict_annotation['image_id']

                if image_id not in update_instances:
                    update_instances[image_id] = {
                        'width': predict_annotation['width'], 'height': predict_annotation['height'], 'annotations': []}

                update_bboxes = update_instances[image_id]['annotations']

                update_bboxes.append({'category_id': predict_annotation['category_id'],
                                      'bbox': predict_annotation['bbox']})

util.write_json_file(
    update_instances, config.project_dir + 'update/instances.json')
