import json

import config
import util

# TODO: Set the paths

instances_file_path_list = [config.project_dir+'easy_gt/instances.json',
                            config.project_dir+'easy_gt/instances_2.json']

merge_instances_file_path = config.project_dir+'easy_gt/instances.json'

###########################################################################

merge_instances = {}

for instances_file_path in instances_file_path_list:

    with open(instances_file_path) as instances_file:
        instances = json.load(instances_file)

        for image_id, instance in instances.items():
            print(image_id)

            if image_id not in merge_instances:
                merge_instances[image_id] = {
                    'width': instance['width'], 'height': instance['height'], 'annotations': []}

            annotations = instance['annotations']
            merge_annotations = merge_instances[image_id]['annotations']

            for annotation in annotations:
                merge_annotations.append(annotation)

util.write_json_file(merge_instances, merge_instances_file_path)
