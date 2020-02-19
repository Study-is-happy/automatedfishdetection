from PIL import Image
import os
import json

import config
import util

# TODO: Set the dir

crop_dir = config.project_dir+'update/'

###########################################################################

black_border = 2

crop_images_dir = crop_dir+'images/'

for image_file_name in os.listdir(crop_images_dir):
    image_file_path = crop_images_dir+image_file_name
    image = Image.open(image_file_path)
    width, height = image.size
    image = image.crop((black_border, black_border, width -
                        black_border, height-black_border))
    # shorter_edge = min(width, height)
    # ratio = 512/shorter_edge
    # image = image.resize((int(width*ratio), int(height*ratio)))
    image.save(image_file_path, "JPEG")

crop_instances_file_path = crop_dir+'instances.json'

if os.path.exists(crop_instances_file_path):
    with open(crop_instances_file_path) as instances_file:
        instances = json.load(instances_file)

    for _, instance in instances.items():
        orig_width = instance['width']
        orig_height = instance['height']
        width = orig_width-black_border*2
        height = orig_height-black_border*2
        instance['width'] = width
        instance['height'] = height
        for annotation in instance['annotations']:
            bbox = annotation['bbox']
            bbox[0] = bbox[0]*orig_width-black_border
            bbox[1] = bbox[1]*orig_height-black_border
            bbox[2] = bbox[2]*orig_width-black_border
            bbox[3] = bbox[3]*orig_height-black_border
            util.norm_bbox(bbox, width, height)
            bbox[0] /= width
            bbox[1] /= height
            bbox[2] /= width
            bbox[3] /= height
            annotation['bbox'] = bbox

    util.write_json_file(instances, crop_instances_file_path)
