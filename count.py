import os
import json

import config

# TODO: Set the path

instances_file_path = config.project_dir + 'train/instances_1.json'

###########################################################################

print_results = {}
for category in config.categories:

    print_results[category] = 0

image_count = 0

with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)
    for _, instance in instances.items():
        image_count += 1

        for annotation in instance['annotations']:
            # if annotation['category_id'] < 2:
            print_results[config.categories[annotation['category_id']]] += 1

print(print_results)
print('images: ' + str(image_count))
