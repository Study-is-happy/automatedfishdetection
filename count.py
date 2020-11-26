import os
import json

import config

# TODO: Set the path

instances_file_path = config.project_dir+'seagate/instances.json'

###########################################################################

print_results = {'Corals': 0, 'Sponges': 0,
                 'Invertebrates': 0, 'Roundfish': 0, 'Skates/Sharks': 0, 'Rockfish': 0, 'Flatfish': 0, 'Other': 0, 'Skates': 0, 'images': 0}


with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)
    for _, instance in instances.items():
        print_results['images'] += 1

        for annotation in instance['annotations']:
            print_results[config.seagate_categories[annotation['category_id']]] += 1

print(print_results)
