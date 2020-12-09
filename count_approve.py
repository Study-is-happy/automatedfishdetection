import csv
import json
import os

import util
import config

# TODO: Set the name list

results_name_list = ['results_1']

###########################################################################

print_results = {'approve': 0, 'reject': 0}

for results_name in results_name_list:
    results_approve_path = config.project_dir + \
        'results/'+results_name+'_approve.csv'

    with open(results_approve_path) as results_approve_file:

        results = csv.reader(results_approve_file)

        headers = next(results)

        for result in results:

            if result[-6] == 'x':
                print_results['approve'] += 1
            else:
                print_results['reject'] += 1

print(print_results)
