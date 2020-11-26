import json
import os
import shutil

import util
import config

# TODO: Set the dirs

src_dataset_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/'

des_dataset_dir = config.project_dir+'seagate/'

###########################################################################

des_instances_file_path = des_dataset_dir+'instances.json'

if os.path.exists(des_instances_file_path):

    with open(des_instances_file_path) as instances_file:
        instances = json.load(instances_file)
else:
    instances = {}

categories = set()


def convert_fct_file(file_path):

    with open(file_path) as annotation_file:

        for annotation_line in annotation_file.readlines():
            annotation_line = annotation_line.strip()
            if annotation_line:
                annotation_line = annotation_line.split(',')

                if len(annotation_line) < 18:
                    pass
                    # print(file_path)
                    # print(annotation_line)

                elif annotation_line[13] != '':

                    image_id = os.path.splitext(annotation_line[3])[0]

                    if image_id not in instances:
                        instances[image_id] = {
                            'width': int(annotation_line[7]), 'height': int(annotation_line[8]), 'annotations': []}

                    instance = instances[image_id]

                    category = annotation_line[10]

                    annotation = {
                        'category_id': config.seagate_categories.index(category)}

                    categories.add(category)

                    annotation['bbox'] = [float(annotation_line[13]), float(annotation_line[14]),
                                          float(annotation_line[13]), float(annotation_line[14])]

                    util.abs_to_rel(
                        annotation['bbox'], instance['width'], instance['height'])

                    instance['annotations'].append(annotation)


for root_path, dir_list, file_list in os.walk(src_dataset_dir):

    if 'backup' not in root_path:

        print(root_path)

        for file_name in file_list:
            if file_name.endswith('fct'):
                convert_fct_file(os.path.join(root_path, file_name))

print(categories)
# util.write_json_file(instances, des_instances_file_path)
