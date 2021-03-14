import json
import csv

import utils
import config

with open(config.project_dir + 'train/instances.json') as instances_file:
    instances = json.load(instances_file)

annotation_index = 0
image_index = 0

with open(config.project_dir + 'viame.csv', 'w') as viame_file:

    writer = csv.writer(viame_file)

    writer.writerow(['# 1: Detection or Track-id,  2: Video or Image Identifier,  3: Unique Frame Identifier,  4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y),  8: Detection or Length Confidence,  9: Fish Length (0 or -1 if invalid),  10-11+: Repeated Species, Confidence Pairs or Attributes'])
    writer.writerow(['# Written on: Wed Feb  5 14:26:30 2020   by: write_detected_object_set_viame_csv', '', '', '', '', '', '', '', '', '', ''])

    for image_id, instance in instances.items():

        width = instance['width']
        height = instance['height']

        for annotation in instance['annotations']:

            bbox = annotation['bbox']

            utils.rel_to_abs(bbox, width, height)

            writer.writerow([annotation_index, image_id + '.jpg', image_index, bbox[0], bbox[1], bbox[2], bbox[3], 1, 0, config.categories[annotation['category_id']], 1])

            annotation_index += 1

    image_index += 1
