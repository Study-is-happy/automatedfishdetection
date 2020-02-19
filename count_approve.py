import csv
import json
import os

import util
import config

# TODO: Set the name list

results_name_list = ['1_1',
                     '1_2',
                     '1_3',
                     '1_4',
                     '1_5',
                     '1_6',
                     '1_7',
                     '1_8',
                     '1_9',
                     '1_10',
                     '1_11',
                     '1_12',
                     '1_13']

###########################################################################

print_results = {'approve': 0, 'reject': 0}

for results_name in results_name_list:
    results_approve_path = config.project_dir + \
        'results/results'+results_name+'_approve.csv'

    with open(results_approve_path) as results_approve_file:

        results = csv.reader(results_approve_file)

        headers = next(results)

        for result in results:

            if result[-6] == 'x':
                print_results['approve'] += 1
            else:
                print_results['reject'] += 1

print(print_results)
