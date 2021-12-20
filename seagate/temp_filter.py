import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import config

all_match_points_dir = config.calib_dir + config.calib_sub_dir + 'all_match_points/'

for match_points_file in sorted(os.listdir(all_match_points_dir)):

    match_points = np.load(all_match_points_dir + match_points_file)

    left_points = match_points['left_points']
    right_points = match_points['right_points']

    if len(left_points) > 1000:

        np.savez(all_match_points_dir + match_points_file,
                 left_points=left_points,
                 right_points=right_points)
