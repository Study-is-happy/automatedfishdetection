import json
import os
import shutil

import util
import config

# TODO: Set the dirs

src_dataset_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/'
# src_dataset_dir = '/home/zhiyongzhang/datasets/seagate/'

des_dataset_dir = config.project_dir+'seagate/'

###########################################################################

des_instances_file_path = des_dataset_dir+'instances.json'

instances = {}

categories = set()

for root_path, dir_list, file_list in os.walk(src_dataset_dir):

    if 'backup' not in root_path:

        # print(root_path)

        for file_name in file_list:

            if file_name.endswith('.fct'):
                with open(os.path.join(root_path, file_name)) as annotation_file:

                    for annotation_line in annotation_file.readlines():
                        annotation_line = annotation_line.strip()
                        if annotation_line:
                            annotation_line = annotation_line.split(',')

                            if len(annotation_line) < 18:
                                pass
                                # print(file_path)
                                # print(annotation_line)
                                # print('---')

                            elif annotation_line[13] != '':

                                image_id = os.path.splitext(
                                    annotation_line[3])[0]

                                if image_id not in instances:
                                    instances[image_id] = {
                                        'width': int(annotation_line[7]), 'height': int(annotation_line[8]), 'annotations': []}

                                instance = instances[image_id]

                                category = annotation_line[10]

                                annotation = {
                                    'category_id': config.categories.index(category)}

                                categories.add(category)

                                annotation['bbox'] = [float(annotation_line[13]), float(annotation_line[14]),
                                                      float(annotation_line[13]), float(annotation_line[14])]

                                util.abs_to_rel(
                                    annotation['bbox'], instance['width'], instance['height'])

                                instance['annotations'].append(annotation)


for root_path, dir_list, file_list in os.walk(src_dataset_dir):

    for file_name in file_list:
        split_file_name = os.path.splitext(file_name)
        if split_file_name[0] in instances:
            if split_file_name[1] == '.jpg':
                shutil.copy(os.path.join(root_path, file_name),
                            des_dataset_dir+'images/')

print(categories)
util.write_json_file(instances, des_instances_file_path)
