import numpy as np
import csv
# import json
import matplotlib.pyplot as plt

import config

metrics_path = config.project_dir+'outputs/metrics.json'
metrics_no_background = config.project_dir+'outputs/metrics_no_background.json'


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

    return np.array(iteration_list), [np.array(mAP_list), np.array(precision_fish_list), np.array(precision_sponge_list), np.array(precision_starfish_list)], [np.array(mAP_list), np.array(recall_fish_list), np.array(recall_sponge_list), np.array(recall_starfish_list)]


iteration_array, precision_arrays, recall_arrays = get_metrics_arrays(
    metrics_path)
iteration_array_no_background, precision_no_background_arrays, recall_no_background_arrays = get_metrics_arrays(
    metrics_no_background)


def plot_arrays(iteration_array, metrics_arrays, metrics_no_background_arrays, titles):
    for index, (metrics_array, metrics_no_background_array, title) in enumerate(zip(metrics_arrays, metrics_no_background_arrays, titles)):
        plt.subplot('22'+str(index+1))
        plt.title(title)
        plt.plot(iteration_array, metrics_array, label='with background')
        plt.plot(iteration_array,
                 metrics_no_background_array, label='without background')
        plt.legend()

    plt.show()


plot_arrays(iteration_array, precision_arrays, precision_no_background_arrays, [
            'mAP', 'precision: fish', 'precision: sponge', 'precision: starfish'])

plot_arrays(iteration_array, recall_arrays, recall_no_background_arrays, [
            'mAP', 'recall: fish', 'recall: sponge', 'recall: starfish'])
