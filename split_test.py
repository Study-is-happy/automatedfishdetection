import os
import json
import shutil

import config
import util


with open(config.project_dir+'train (copy)/instances.json') as instances_file:

    instances = json.load(instances_file)
    test_instances = {}
    train_instances = {}

    count = 0

    for image_id, instance in instances.items():

        if len(instance['annotations']) > 1 and len(instance['annotations']) < 4 and count < 300:

            shutil.copy(config.project_dir+'train (copy)/images/'+image_id +
                        '.jpg', config.project_dir+'test/images/')

            test_instances[image_id] = instance
            count += 1
        else:
            shutil.copy(config.project_dir+'train (copy)/images/'+image_id +
                        '.jpg', config.project_dir+'train/images/')
            train_instances[image_id] = instance

    util.write_json_file(
        test_instances, config.project_dir+'test/instances.json')
    util.write_json_file(
        train_instances, config.project_dir+'train/instances.json')
