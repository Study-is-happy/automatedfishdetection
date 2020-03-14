from xml.etree import ElementTree
import os

import config

annotations_dir = config.project_dir+'pro/annotations (copy)/'

for annotation_file_name in os.listdir(annotations_dir):

    annotation_file_path = annotations_dir+annotation_file_name

    annotation_node = ElementTree.parse(annotation_file_path)

    object_nodes = annotation_node.findall('object')

    for object_node in object_nodes:

        if object_node.find('name').text.lower() == 'unknown':
            break

    else:
        os.remove(annotation_file_path)
