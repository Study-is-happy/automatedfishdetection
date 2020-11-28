from xml.etree import ElementTree
import shutil
import os

import util
import config

# TODO: Set the dirs

src_dataset_dir = '/home/zhiyongzhang/datasets/Fish_Training_Data/Training/'

src_image_dir = src_dataset_dir+'Pacstorm/images/'

src_annotation_dir = src_dataset_dir+'Pacstorm_Annotations/'

# src_dataset_dir = config.project_dir+'easy_gt (copy)/'

# src_image_dir = src_dataset_dir+'images/'

# src_annotation_dir = src_dataset_dir+'annotations/'

des_dataset_dir = config.project_dir+'datasets/'

###########################################################################

des_image_dir = des_dataset_dir+'images/'

des_image_file_path = des_dataset_dir+'images.txt'

des_annotation_dir = des_dataset_dir+'annotations/'


def get_int_node(parent_node, node_name):

    return int(parent_node.find(node_name).text)


def get_string_node(parent_node, node_name):
    node = parent_node.find(node_name)
    if node is None:
        return 'Unspecified'
    return node.text


image_names = os.listdir(src_image_dir)

for image_name in image_names:

    src_image_path = src_image_dir+image_name

    image_id = os.path.splitext(image_name)[0]

    annotation_node = ElementTree.parse(
        src_annotation_dir+image_id+'.xml').getroot()

    size_node = annotation_node.find('size')

    width = get_int_node(size_node, 'width')
    height = get_int_node(size_node, 'height')

    object_nodes = annotation_node.findall('object')

    bboxes = []

    for object_node in object_nodes:

        category = get_string_node(object_node, 'name')

        if category in config.categories:

            bndbox_node = object_node.find('bndbox')

            xmin = get_int_node(bndbox_node, 'xmin')
            ymin = get_int_node(bndbox_node, 'ymin')
            xmax = get_int_node(bndbox_node, 'xmax')
            ymax = get_int_node(bndbox_node, 'ymax')

            bbox = util.update_bbox(category, xmin, ymin,
                                    xmax, ymax)

            bboxes.append(bbox)

    if len(bboxes) > 0:

        shutil.copy(src_image_path, des_image_dir)
        util.write_update_file(des_image_dir, des_annotation_dir, image_id,
                               width, height, bboxes)
        util.merge_image_id(des_image_file_path, image_id)
