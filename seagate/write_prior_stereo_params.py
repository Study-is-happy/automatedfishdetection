import numpy as np

import config

stereo_params = np.load(config.calib_dir + 'd20191103_18/stereo_params.npz')

np.savez(config.calib_dir + config.calib_sub_dir + 'prior_stereo_params',
         rotation_matrix=stereo_params['rotation_matrix'],
         translation=stereo_params['translation'])
