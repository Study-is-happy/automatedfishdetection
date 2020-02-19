from xml.dom import minidom
import json


xml_document = minidom.Document()

annotation_node = xml_document.createElement('annotation')
xml_document.appendChild(annotation_node)

folder_node = xml_document.createElement('folder')
annotation_node.appendChild(folder_node)
folder_node_text = xml_document.createTextNode(images_dir)
folder_node.appendChild(folder_node_text)

filename_node = xml_document.createElement('filename')
annotation_node.appendChild(filename_node)
filename_node_text = xml_document.createTextNode(image_name+'.jpg')
filename_node.appendChild(filename_node_text)

size_node = xml_document.createElement('size')
annotation_node.appendChild(size_node)

width_node = xml_document.createElement('width')
size_node.appendChild(width_node)
width_node_text = xml_document.createTextNode(str(width))
width_node.appendChild(width_node_text)

height_node = xml_document.createElement('height')
size_node.appendChild(height_node)
height_node_text = xml_document.createTextNode(str(height))
height_node.appendChild(height_node_text)

for bbox in bboxes:

object_node = xml_document.createElement('object')
annotation_node.appendChild(object_node)

name_node = xml_document.createElement('name')
object_node.appendChild(name_node)
name_node_text = xml_document.createTextNode(bbox.label)
name_node.appendChild(name_node_text)

pose_node = xml_document.createElement('pose')
object_node.appendChild(pose_node)
pose_node_text = xml_document.createTextNode(bbox.pose)
pose_node.appendChild(pose_node_text)

truncated_node = xml_document.createElement('truncated')
object_node.appendChild(truncated_node)
truncated_node_text = xml_document.createTextNode(str(bbox.truncated))
truncated_node.appendChild(truncated_node_text)

difficult_node = xml_document.createElement('difficult')
object_node.appendChild(difficult_node)
difficult_node_text = xml_document.createTextNode(str(bbox.difficult))
difficult_node.appendChild(difficult_node_text)

temp_node = xml_document.createElement('temp')
object_node.appendChild(temp_node)
temp_node_text = xml_document.createTextNode(str(bbox.temp))
temp_node.appendChild(temp_node_text)

bndbox_node = xml_document.createElement('bndbox')
object_node.appendChild(bndbox_node)

xmin_node = xml_document.createElement('xmin')
bndbox_node.appendChild(xmin_node)
xmin_node_text = xml_document.createTextNode(str(bbox.xmin))
xmin_node.appendChild(xmin_node_text)

ymin_node = xml_document.createElement('ymin')
bndbox_node.appendChild(ymin_node)
ymin_node_text = xml_document.createTextNode(str(bbox.ymin))
ymin_node.appendChild(ymin_node_text)

xmax_node = xml_document.createElement('xmax')
bndbox_node.appendChild(xmax_node)
xmax_node_text = xml_document.createTextNode(str(bbox.xmax))
xmax_node.appendChild(xmax_node_text)

ymax_node = xml_document.createElement('ymax')
bndbox_node.appendChild(ymax_node)
ymax_node_text = xml_document.createTextNode(str(bbox.ymax))
ymax_node.appendChild(ymax_node_text)

with open(annotations_dir+image_name+'.xml', 'w', encoding='utf-8') as annotation_file:
xml_document.writexml(
    annotation_file, addindent='\t', newl='\n', encoding='UTF-8')