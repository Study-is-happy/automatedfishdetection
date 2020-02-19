import os
import json
import shutil

import config
import util

with open(config.project_dir+'update/instances.json') as instances_file:
    instances = json.load(instances_file)

for image_id in instances:

    shutil.copy(config.project_dir+'update/images/'+image_id +
                '.jpg', config.project_dir+'update/unannotated/')
