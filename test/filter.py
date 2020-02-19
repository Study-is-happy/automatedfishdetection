import config
import util

import os

dataset_dir = config.update_dir

image_dir = dataset_dir+'image/'

annotation_dir = dataset_dir+'annotation/'

image_file_path = dataset_dir+'image.txt'

image_names = util.read_image_file(image_file_path)

util.create_file(image_file_path)

for image_name in image_names:

    annotation = util.read_annotation_file(
        annotation_dir+image_name+'.xml')

    bboxes = []

    for bbox in annotation['bboxes']:
        if bbox.adjust == 0:
            bboxes.append(bbox)

    if len(bboxes) == 0:
        os.remove(annotation_dir+image_name+'.xml')
        os.remove(image_dir+image_name+'.jpg')

    else:
        util.merge_image_name(image_file_path, image_name)
        util.write_xml(image_dir, annotation_dir, image_name,
                       annotation['width'], annotation['height'], bboxes)
