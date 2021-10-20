import numpy as np
from scipy.spatial.transform import Rotation

import config

stereo_params = np.load(config.calib_dir + config.calib_sub_dir + 'stereo_params.npz')

print(Rotation.from_matrix(stereo_params['rotation_matrix']).as_euler('zyx', degrees=True))
print(stereo_params['translation'])
