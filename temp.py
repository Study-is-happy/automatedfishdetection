import os
import json
import shutil

import config
import util

# TODO: Set the dirs

dataset_dir = config.project_dir+'easy_gt/'

###########################################################################

src_images_dir = config.project_dir+'raw/images/'
instances_file_path = dataset_dir+'instances.json'

with open(instances_file_path) as instances_file:
    instances = json.load(instances_file)

for image_id in instances.keys():

    shutil.copy(src_images_dir+image_id+'.jpg', dataset_dir+'images/')
