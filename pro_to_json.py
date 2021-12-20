import matplotlib.image as mpimg
import shutil
import json
import os

import utils
import config

# TODO: Set the dirs

src_annotations_dir = config.project_dir + 'pro_annotations/'

src_images_dir = config.project_dir + 'update/images/'

des_instances_file_path = config.project_dir + 'update/instances_1.json'

###########################################################################

if os.path.exists(des_instances_file_path):

    with open(des_instances_file_path) as instances_file:
        instances = json.load(instances_file)
else:
    instances = {}

for annotation_file_name in os.listdir(src_annotations_dir):

    print(annotation_file_name)

    with open(src_annotations_dir + annotation_file_name) as pro_instance_file:
        pro_instance = json.load(pro_instance_file)

    image_id = os.path.splitext(annotation_file_name)[0]

    image = mpimg.imread(src_images_dir + pro_instance['originStorageName'])

    instances[image_id] = {'width': image.shape[1], 'height': image.shape[0], 'annotations': []}

    instance = instances[image_id]

    instance['width'] = image.shape[1]
    instance['height'] = image.shape[0]

    instance['annotations'] = []

    for pro_annotation in pro_instance['LabelDetail']['objects']:

        category = pro_annotation['labelName']

        if category not in config.categories:
            print(category)

        annotation = {'category_id': config.categories.index(category)}

        pro_bbox = pro_annotation['feature']['points']

        annotation['bbox'] = [float(pro_bbox[0]['x']),
                              float(pro_bbox[0]['y']),
                              float(pro_bbox[2]['x']),
                              float(pro_bbox[2]['y'])]

        utils.abs_to_rel(
            annotation['bbox'], instance['width'], instance['height'])

        instance['annotations'].append(annotation)

utils.write_json_file(instances, des_instances_file_path)
