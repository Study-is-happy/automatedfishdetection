from xml.etree import ElementTree
import os

import config

dataset_dir = config.project_dir+'pro (copy)/'

images_dir = dataset_dir+'images/'

annotations_dir = dataset_dir+'annotations/'

for annotation_file_name in os.listdir(annotations_dir):

    image_id = os.path.splitext(annotation_file_name)[0]

    annotation_file_path = annotations_dir+image_id+'.xml'

    annotation_node = ElementTree.parse(annotation_file_path)

    object_nodes = annotation_node.findall('object')

    for object_node in object_nodes:

        if object_node.find('name').text.lower() == 'unknown':
            break

    else:
        os.remove(images_dir+image_id+'.jpg')
        os.remove(annotation_file_path)
