import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import config

stereo_params_dir = config.calib_dir + config.calib_sub_dir + 'stereo_params/'

plot_list = []

for stereo_params_file in sorted(os.listdir(stereo_params_dir)):

    stereo_params = np.load(stereo_params_dir + stereo_params_file)

    rotation_matrix = stereo_params['rotation_matrix']
    rotation = Rotation.from_matrix(rotation_matrix).as_euler('zyx', degrees=True)

    translation = stereo_params['translation']

    # translation_list.append(translation[2])
    plot_list.append(translation)

plot_list = np.array(plot_list)

plt.plot(plot_list[:, 0])
plt.plot(plot_list[:, 1])
plt.plot(plot_list[:, 2])
plt.show()
