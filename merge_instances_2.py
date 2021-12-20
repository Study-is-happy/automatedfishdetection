import json
import os

import config
import util

# TODO: Set the paths

instances_file_path_list = [config.project_dir + 'update/rockfish_instances.json',
                            config.project_dir + 'update/pmfs_instances.json']

merge_instances_file_path = config.project_dir + 'update/instances.json'

###########################################################################

merge_instances = {}

for instances_file_path in instances_file_path_list:

    with open(instances_file_path) as instances_file:
        instances = json.load(instances_file)

        for image_id, instance in instances.items():

            if os.path.exists(config.project_dir + 'update/images/' + image_id + '.jpg'):

                if image_id not in merge_instances:
                    merge_instances[image_id] = {
                        'width': instance['width'], 'height': instance['height'], 'annotations': []}

                annotations = instance['annotations']
                merge_annotations = merge_instances[image_id]['annotations']

                for annotation in annotations:
                    for merge_annotation in merge_annotations:
                        if util.get_bboxes_iou(annotation['bbox'], merge_annotation['bbox']) > 0.5:
                            break
                    else:
                        merge_annotations.append(annotation)

            else:
                print(image_id)

util.write_json_file(merge_instances, merge_instances_file_path)
