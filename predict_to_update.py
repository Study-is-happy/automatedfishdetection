import json

import config
import util

easy_gt_annotation_indexes = [9]

with open(config.project_dir+'update/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)

with open(config.project_dir+'predict/annotation_ids.csv') as annotations_file_names:
    next(annotations_file_names)

    for annotations_file_name in annotations_file_names:
        annotations_file_name = annotations_file_name.rstrip('\n')

        with open(config.project_dir+'predict/annotations/'+annotations_file_name+'.json') as predict_annotations_file:
            predict_annotations = json.load(predict_annotations_file)

            for index, predict_annotation in enumerate(predict_annotations):
                if index not in easy_gt_annotation_indexes:

                    image_id = predict_annotation['image_id']

                    if image_id not in update_instances:
                        update_instances[image_id] = {
                            'width': predict_annotation['width'], 'height': predict_annotation['height'], 'annotations': []}

                    update_bboxes = update_instances[image_id]['annotations']

                    category_id = config.categories.index('background')
                    update_bboxes.append({'category_id': -1,
                                          'bbox': predict_annotation['bbox']})

util.write_json_file(
    update_instances, config.project_dir+'update/instances.json')
