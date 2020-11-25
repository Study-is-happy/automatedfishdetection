import json
import os
import shutil

import util
import config

# TODO: Set the dirs

src_dataset_dir = '/media/zhiyongzhang/Seagate Desktop Drive/AUV_images_fcts/'

des_dataset_dir = config.project_dir+'seagate/'

###########################################################################

# des_images_dir = des_dataset_dir+'images/'

des_instances_file_path = des_dataset_dir+'instances.json'

if os.path.exists(des_instances_file_path):

    with open(des_instances_file_path) as instances_file:
        instances = json.load(instances_file)
else:
    instances = {}

categories = set()


def traverse_dir(dir_path):
    for root_path, dirs, files in os.walk(dir_path):

        for dir_name in dirs:
            if 'backup' not in dir_name:
                traverse_dir(os.path.join(root_path, dir_name))

        for file_name in files:
            if file_name.endswith('fct'):
                convert_fct_file(os.path.join(root_path, file_name))


test_count = 0


def convert_fct_file(file_path):
    with open(file_path) as annotation_file:

        for annotation_line in annotation_file.readlines():
            annotation_line = annotation_line.strip()
            if annotation_line:
                annotation_line = annotation_line.split(',')

                if len(annotation_line) < 18:
                    # print(file_path)
                    print(annotation_line)

                elif annotation_line[13] != '':

                    image_id = os.path.splitext(annotation_line[3])[0]

                    if image_id not in instances:
                        instances[image_id] = {
                            'width': int(annotation_line[7]), 'height': int(annotation_line[8]), 'annotations': []}

                    instance = instances[image_id]

                    annotation = {'category_id': annotation_line[10]}

                    categories.add(annotation_line[10])

                    annotation['bbox'] = [float(annotation_line[13]), float(annotation_line[14]),
                                          float(annotation_line[13]), float(annotation_line[14])]

                    util.abs_to_rel(
                        annotation['bbox'], instance['width'], instance['height'])

                    instance['annotations'].append(annotation)

                    global test_count
                    test_count += 1
                    print(test_count)


traverse_dir(src_dataset_dir)

print(categories)
util.write_json_file(instances, des_instances_file_path)

# if os.path.exists(des_instances_file_path):

#     with open(des_instances_file_path) as instances_file:
#         instances = json.load(instances_file)
# else:
#     instances = {}

# for annotation_file_name in os.listdir(src_annotations_dir):

#     image_id = os.path.splitext(annotation_file_name)[0]

#     instance = {}

#     annotation_node = ElementTree.parse(
#         src_annotations_dir+annotation_file_name)

#     size_node = annotation_node.find('size')

#     instance['width'] = int(size_node.find('width').text)
#     instance['height'] = int(size_node.find('height').text)

#     object_nodes = annotation_node.findall('object')

#     instance['annotations'] = []

#     for object_node in object_nodes:

#         category = object_node.find('name').text.lower()

#         if category in config.categories:

#             annotation = {'category_id': config.categories.index(category)}

#             bndbox_node = object_node.find('bndbox')

#             annotation['bbox'] = [float(bndbox_node.find('xmin').text)-1,
#                                   float(bndbox_node.find('ymin').text)-1,
#                                   float(bndbox_node.find('xmax').text),
#                                   float(bndbox_node.find('ymax').text)]

#             util.abs_to_rel(
#                 annotation['bbox'], instance['width'], instance['height'])

#             # annotation['difficult'] = int(
#             #     object_node.find('difficult').text)
#             annotation['difficult'] = 1

#             # annotation['temp'] = 0

#             instance['annotations'].append(annotation)

#     if len(instance['annotations']) > 0:

#         shutil.copy(src_images_dir+image_id+'.jpg', des_images_dir)

#         instances[image_id] = instance

# util.write_json_file(instances, des_instances_file_path)
