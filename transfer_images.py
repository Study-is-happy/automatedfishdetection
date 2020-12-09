import json
import shutil

import config
import util

# TODO: Set the dirs

src_images_dir = config.project_dir+'train/images/'

des_dataset_dir = config.project_dir+'easy_gt/'

###########################################################################

with open(des_dataset_dir + 'instances.json') as instances_file:
    instances = json.load(instances_file)

    for image_id in instances.keys():
        shutil.copy(src_images_dir+image_id+'.jpg', des_dataset_dir+'images/')
