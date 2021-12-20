import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import utm as utm_tf

import csv
import config


def get_array(str_array):
    return np.double(str_array[1:-1].strip().split())


ax = plt.axes(projection='3d')

xyz_list = []

with open(config.project_dir + 'rockfish_size.csv') as rockfish_file:

    rockfish_reader = csv.reader(rockfish_file)
    for rockfish in rockfish_reader:

        latlng = get_array(rockfish[3])

        utm = np.array(utm_tf.from_latlon(latlng[0], latlng[1])[0:2])

        utm_xyz = np.array([utm[0], utm[1], 0])

        xyz = np.array([get_array(rockfish[4]),
                        get_array(rockfish[5]),
                        get_array(rockfish[6]),
                        get_array(rockfish[7])]) / 100. + utm_xyz

        xyz[:, 2] = -xyz[:, 2]

        xyz_list.extend(xyz)

        ax.add_collection3d(Poly3DCollection([xyz], edgecolor='red', facecolor='white'))

xyz_list = np.array(xyz_list)

ax.set_xlim3d(np.min(xyz_list[:, 0]), np.max(xyz_list[:, 0]))
ax.set_ylim3d(np.min(xyz_list[:, 1]), np.max(xyz_list[:, 1]))
ax.set_zlim3d(np.min(xyz_list[:, 2]), np.max(xyz_list[:, 2]))

# ax.set_box_aspect(aspect=(1, 1, 0.01))

plt.show()
