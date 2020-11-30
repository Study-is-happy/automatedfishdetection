from xml.etree import ElementTree
import shutil
import json
import os

import util
import config

# TODO: Set the dirs

# src_dataset_dir = '/home/zhiyongzhang/datasets/201009_PacStorm/d20100923_1/'

# src_images_dir = src_dataset_dir+'down_images/'

# src_annotations_dir = src_dataset_dir+'down_Annotations/'

src_dataset_dir = config.project_dir+'pro/'

src_images_dir = src_dataset_dir+'images/'

src_annotations_dir = src_dataset_dir+'annotations/'

des_dataset_dir = config.project_dir+'update/'

###########################################################################

des_images_dir = des_dataset_dir+'images/'

des_instances_file_path = des_dataset_dir+'instances.json'

if os.path.exists(des_instances_file_path):

    with open(des_instances_file_path) as instances_file:
        instances = json.load(instances_file)
else:
    instances = {}

for annotation_file_name in os.listdir(src_annotations_dir):

    image_id = os.path.splitext(annotation_file_name)[0]

    instance = {}

    annotation_node = ElementTree.parse(
        src_annotations_dir+annotation_file_name)

    size_node = annotation_node.find('size')

    instance['width'] = int(size_node.find('width').text)
    instance['height'] = int(size_node.find('height').text)

    object_nodes = annotation_node.findall('object')

    instance['annotations'] = []

    for object_node in object_nodes:

        category = object_node.find('name').text.lower()

        if category in config.categories:

            annotation = {'category_id': config.categories.index(category)}

            bndbox_node = object_node.find('bndbox')

            annotation['bbox'] = [float(bndbox_node.find('xmin').text)-1,
                                  float(bndbox_node.find('ymin').text)-1,
                                  float(bndbox_node.find('xmax').text),
                                  float(bndbox_node.find('ymax').text)]

            util.abs_to_rel(
                annotation['bbox'], instance['width'], instance['height'])

            # annotation['difficult'] = int(
            #     object_node.find('difficult').text)

            instance['annotations'].append(annotation)

    if len(instance['annotations']) > 0:

        shutil.copy(src_images_dir+image_id+'.jpg', des_images_dir)

        instances[image_id] = instance

util.write_json_file(instances, des_instances_file_path)
