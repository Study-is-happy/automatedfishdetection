import os
import json

import config

# TODO: Set the path

instances_file_path = config.project_dir+'seagate/instances.json'

###########################################################################

print_results = {'images': 0}
for category in config.categories:
    print_results[category] = 0

with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)
    for _, instance in instances.items():
        print_results['images'] += 1

        for annotation in instance['annotations']:
            print_results[config.categories[annotation['category_id']]] += 1

print(print_results)
