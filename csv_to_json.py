import csv
import json
import os
import cv2

import util
import config

dataset_dir = config.project_dir+'lynker/'

with open(dataset_dir+'detections.csv') as detections_csv_file:

    detections = csv.reader(detections_csv_file)

    headers = next(detections)

    instances = {}

    for detection in detections:

        image_file_name = detection[1]

        image_id = os.path.splitext(image_file_name)[0]

        if image_id not in instances:

            image = cv2.imread(dataset_dir+'images/'+image_file_name)

            image_shape = image.shape

            instances[image_id] = {'width': image_shape[1],
                                   'height': image_shape[0], 'annotations': []}

        instance = instances[image_id]

        annotation = {
            'category_id': config.lynker_categories.index(detection[-2])}

        annotation['bbox'] = [float(detection[3]),
                              float(detection[4]), float(detection[5]), float(detection[6])]
        util.abs_to_rel(
            annotation['bbox'], instance['width'], instance['height'])

        instance['annotations'].append(annotation)


with open(dataset_dir+'instances.json', 'w') as instances_json_file:

    json.dump(instances, instances_json_file, indent='\t')
