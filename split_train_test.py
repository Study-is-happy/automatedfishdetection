import json
import shutil

import config
import utils

test_instances = {}
train_instances = {}

with open(config.project_dir + 'all/instances.json') as instances_file:

    instances = json.load(instances_file)

    test_count = 0

    for image_id, instance in instances.items():

        print(image_id)

        annotation_count = 0

        for annotation in instance['annotations']:
            if annotation['category_id'] == 0:
                annotation_count += 1

        if annotation_count >= 1 and annotation_count < 4 and test_count < 5000:

            shutil.copy(config.project_dir + 'all/images/' + image_id +
                        '.jpg', config.project_dir + 'test/images/')

            test_instances[image_id] = instance
            test_count += 1
        else:
            shutil.copy(config.project_dir + 'all/images/' + image_id +
                        '.jpg', config.project_dir + 'train/images/')
            train_instances[image_id] = instance

utils.write_json_file(
    test_instances, config.project_dir + 'test/instances.json')
utils.write_json_file(
    train_instances, config.project_dir + 'train/instances.json')
