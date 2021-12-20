from xml.dom import minidom
import shutil
import json

import utils
import config

with open(config.project_dir + 'init/instances.json') as update_instances_file:
    update_instances = json.load(update_instances_file)

for image_id, instance in update_instances.items():

    # shutil.copy(config.project_dir + 'predict/images/' + image_id
    #             + '.jpg', config.project_dir + 'xml/images/')

    xml_file_path = config.project_dir + 'init/annotations/' + image_id + '.xml'

    width = instance['width']
    height = instance['height']

    xml_document = minidom.Document()

    annotation_node = xml_document.createElement('annotation')
    xml_document.appendChild(annotation_node)

    folder_node = xml_document.createElement('folder')
    annotation_node.appendChild(folder_node)
    folder_node_text = xml_document.createTextNode('images/')
    folder_node.appendChild(folder_node_text)

    filename_node = xml_document.createElement('filename')
    annotation_node.appendChild(filename_node)
    filename_node_text = xml_document.createTextNode(image_id + '.jpg')
    filename_node.appendChild(filename_node_text)

    path_node = xml_document.createElement('path')
    annotation_node.appendChild(path_node)
    path_node_text = xml_document.createTextNode(
        'images/' + image_id + '.jpg')
    path_node.appendChild(path_node_text)

    source_node = xml_document.createElement('source')
    annotation_node.appendChild(source_node)

    database_node = xml_document.createElement('database')
    source_node.appendChild(database_node)
    database_node_text = xml_document.createTextNode('Unknown')
    database_node.appendChild(database_node_text)

    size_node = xml_document.createElement('size')
    annotation_node.appendChild(size_node)

    width_node = xml_document.createElement('width')
    size_node.appendChild(width_node)
    width_node_text = xml_document.createTextNode(
        str(width))
    width_node.appendChild(width_node_text)

    height_node = xml_document.createElement('height')
    size_node.appendChild(height_node)
    height_node_text = xml_document.createTextNode(
        str(height))
    height_node.appendChild(height_node_text)

    depth_node = xml_document.createElement('depth')
    size_node.appendChild(depth_node)
    depth_node_text = xml_document.createTextNode(
        str(3))
    depth_node.appendChild(depth_node_text)

    segmented_node = xml_document.createElement('segmented')
    annotation_node.appendChild(segmented_node)
    segmented_node_text = xml_document.createTextNode('0')
    segmented_node.appendChild(segmented_node_text)

    for annotation in instance['annotations']:

        object_node = xml_document.createElement('object')
        annotation_node.appendChild(object_node)

        name_node = xml_document.createElement('name')
        object_node.appendChild(name_node)

        if 'score' in annotation:
            name_text = 'predict'
        elif annotation['category_id'] == -1:
            name_text = 'unknown'
        else:
            name_text = config.categories[annotation['category_id']]

        name_node_text = xml_document.createTextNode(name_text)
        name_node.appendChild(name_node_text)

        pose_node = xml_document.createElement('pose')
        object_node.appendChild(pose_node)
        pose_node_text = xml_document.createTextNode('Unspecified')
        pose_node.appendChild(pose_node_text)

        truncated_node = xml_document.createElement('truncated')
        object_node.appendChild(truncated_node)
        truncated_node_text = xml_document.createTextNode('0')
        truncated_node.appendChild(truncated_node_text)

        bndbox_node = xml_document.createElement('bndbox')
        object_node.appendChild(bndbox_node)

        bbox = annotation['bbox']
        utils.pascal_voc_rel_to_abs(bbox, width, height)

        xmin_node = xml_document.createElement('xmin')
        bndbox_node.appendChild(xmin_node)
        xmin_node_text = xml_document.createTextNode(str(bbox[0]))
        xmin_node.appendChild(xmin_node_text)

        ymin_node = xml_document.createElement('ymin')
        bndbox_node.appendChild(ymin_node)
        ymin_node_text = xml_document.createTextNode(str(bbox[1]))
        ymin_node.appendChild(ymin_node_text)

        xmax_node = xml_document.createElement('xmax')
        bndbox_node.appendChild(xmax_node)
        xmax_node_text = xml_document.createTextNode(str(bbox[2]))
        xmax_node.appendChild(xmax_node_text)

        ymax_node = xml_document.createElement('ymax')
        bndbox_node.appendChild(ymax_node)
        ymax_node_text = xml_document.createTextNode(str(bbox[3]))
        ymax_node.appendChild(ymax_node_text)

    with open(xml_file_path, 'w', encoding='utf-8') as xml_file:
        xml_document.writexml(
            xml_file, addindent='\t', newl='\n', encoding='UTF-8')
