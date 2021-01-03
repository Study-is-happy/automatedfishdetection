import json
import os
import shutil
import PIL.Image

import util
import config

# TODO: Set the dirs

src_dataset_dir = '/media/auv/Seagate Desktop Drive/AUV_images_fcts/'

des_dataset_dir = config.project_dir + 'train/'

###########################################################################

des_instances_file_path = des_dataset_dir + 'instances.json'

init_box_size = 1

instances = {}

categories = set()


def is_port_dir(root_path):

    parent_dir_name, dir_name = os.path.split(root_path)

    return (dir_name == 'port_rectified') and ('backup' not in parent_dir_name)


for root_path, dir_list, file_list in os.walk(src_dataset_dir):

    if is_port_dir(root_path):

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

                                category = annotation_line[10].lower()

                                categories.add(category)

                                if category in config.categories:

                                    image_id = os.path.splitext(
                                        annotation_line[3])[0]

                                    if image_id not in instances:
                                        instances[image_id] = {
                                            'width': int(annotation_line[7]), 'height': int(annotation_line[8]), 'annotations': []}

                                    instance = instances[image_id]

                                    annotation = {
                                        'category_id': config.categories.index(category)}

                                    annotation['bbox'] = [float(annotation_line[13]) - init_box_size, instance['height'] - float(annotation_line[14]) - init_box_size,
                                                          float(annotation_line[13]) + init_box_size, instance['height'] - float(annotation_line[14]) + init_box_size]

                                    util.abs_to_rel(
                                        annotation['bbox'], instance['width'], instance['height'])

                                    util.norm_rel_bbox(annotation['bbox'])

                                    instance['annotations'].append(annotation)

print(categories)
util.write_json_file(instances, des_instances_file_path)

des_images_dir = des_dataset_dir + 'images/'
des_empty_images_dir = des_dataset_dir + 'empty_images/'

for root_path, dir_list, file_list in os.walk(src_dataset_dir):

    if is_port_dir(root_path):

        for file_name in file_list:
            image_id, extension = os.path.splitext(file_name)
            file_path = os.path.join(root_path, file_name)
            if extension == '.jpg':
                if image_id in instances:
                    shutil.copy(file_path, des_images_dir)
                # else:
                #     shutil.copy(file_path, des_empty_images_dir)
            elif extension == '.tif':
                image = PIL.Image.open(file_path)
                if image_id in instances:
                    image.save(des_images_dir + image_id + '.jpg', "JPEG")
                # else:
                #     image.save(des_empty_images_dir+image_id+'.jpg', "JPEG")
