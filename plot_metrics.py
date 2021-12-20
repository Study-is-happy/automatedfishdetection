import numpy as np
import csv
import scipy.interpolate
import matplotlib.pyplot as plt

import config

metrics_with_background = config.project_dir + \
    'outputs/metrics_with_background.json'
metrics_no_background = config.project_dir + 'outputs/metrics_no_background.json'


def get_value(metrics):
    return float(metrics.split(': ')[1])


def get_metrics_arrays(metrics_path):

    iteration_list = []
    mAP_list = []
    precision_fish_list = []
    precision_sponge_list = []
    precision_starfish_list = []
    recall_fish_list = []
    recall_sponge_list = []
    recall_starfish_list = []

    with open(metrics_path) as metrics_file:

        metrics = csv.reader(metrics_file)

        for metrics_list in metrics:

            if len(metrics_list) == 24:
                iteration_list.append(get_value(metrics_list[5]))
                mAP_list.append(get_value(metrics_list[11]))
                precision_fish_list.append(get_value(metrics_list[12]))
                precision_sponge_list.append(get_value(metrics_list[13]))
                precision_starfish_list.append(get_value(metrics_list[14]))
                recall_fish_list.append(get_value(metrics_list[15]))
                recall_sponge_list.append(get_value(metrics_list[16]))
                recall_starfish_list.append(get_value(metrics_list[17]))

    # return np.array(iteration_list), np.array(mAP_list), np.array(precision_fish_list), np.array(precision_sponge_list), np.array(precision_starfish_list)
    iteration_list = np.array(iteration_list)

    smooth_mAP_list = get_smooth_y_list(mAP_list)
    smooth_precision_fish_list = get_smooth_y_list(precision_fish_list)
    smooth_precision_sponge_list = get_smooth_y_list(precision_sponge_list)
    smooth_precision_starfish_list = get_smooth_y_list(precision_starfish_list)

    return iteration_list, smooth_mAP_list, smooth_precision_fish_list, smooth_precision_sponge_list, smooth_precision_starfish_list


def get_smooth_y_list(y_list):

    y_list = y_list[:1] + y_list + y_list[-1:]

    return [y_list[0]] + list(np.convolve(y_list, np.ones(5) / 5, 'valid')) + [y_list[-1]]

    # a_BSpline = scipy.interpolate.make_interp_spline(x_list, y_list)
    # return a_BSpline(smooth_x_list)


iteration_list, mAP_list, precision_fish_list, precision_sponge_list, precision_starfish_list = get_metrics_arrays(
    metrics_with_background)
iteration_list_no_bg, mAP_list_no_bg, precision_fish_list_no_bg, precision_sponge_list_no_bg, precision_starfish_list_no_bg = get_metrics_arrays(
    metrics_no_background)


def plot_arrays(iteration_array, metrics_arrays, metrics_no_background_arrays, titles):

    for index, (metrics_array, metrics_no_background_array, title) in enumerate(zip(metrics_arrays, metrics_no_background_arrays, titles)):
        # plt.subplot(220 + (index + 1))
        plt.title(title)

        plt.plot(iteration_array, metrics_array, label='with background')
        plt.plot(iteration_array,
                 metrics_no_background_array, label='without background')
        plt.legend()

        plt.show()


plot_arrays(iteration_list, [mAP_list, precision_fish_list, precision_sponge_list, precision_starfish_list],
            [mAP_list_no_bg, precision_fish_list_no_bg, precision_sponge_list_no_bg, precision_starfish_list_no_bg], [
    'mAP', 'precision: fish', 'precision: sponge', 'precision: starfish'])

# plot_arrays(iteration_array, recall_arrays, recall_no_background_arrays, [
#             'mAP', 'recall: fish', 'recall: sponge', 'recall: starfish'])
