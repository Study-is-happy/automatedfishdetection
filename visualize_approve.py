import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import csv
import os

import config
import util

results_approve_path = config.project_dir + \
    'results_approve/' + config.results_name

predict_dir = config.project_dir+'predict/'

with open(results_approve_path) as results_approve_file:

    results = csv.reader(results_approve_file)

    next(results)

    for result in results:

        # if result[-9] != '12616':
        #     continue

        print(result[-9])

        result_annotations = json.loads(result[-8])

        with open(predict_dir+'annotations/'+result[-9]+'.json') as predict_annotations_file:
            predict_annotations = json.load(predict_annotations_file)

        conf_indexes = json.loads(result[-4])
        not_conf_indexes = json.loads(result[-3])
        approved_gt_indexes = json.loads(result[-2])

        for index, (predict_annotation, result_annotation) in enumerate(zip(predict_annotations, result_annotations)):

            if result[-6] == 'x' or index not in config.gt_indexes:
                continue

            # if result_annotation['image_id'] != '20161027.175242.00310_rect_color':
            #     continue

            if index in conf_indexes+approved_gt_indexes:
                judge_color = 'green'
                judge_text = 'Approve'
            elif index in not_conf_indexes:
                judge_color = 'orange'
                judge_text = 'Not Sure'
            else:
                judge_color = 'red'
                judge_text = 'Reject'

            image = mpimg.imread(predict_dir+'images/' +
                                 result_annotation['image_id']+'.jpg')

            width = image.shape[1]

            height = image.shape[0]

            predict_color = config.colors[predict_annotation['category_id']]
            result_color = config.colors[result_annotation['category_id']]

            predict_bbox = predict_annotation['bbox']
            result_bbox = result_annotation['bbox']

            iou = util.get_bboxes_iou(predict_bbox, result_bbox)

            util.rel_to_abs(predict_bbox, width, height)
            util.rel_to_abs(result_bbox, width, height)

            xmin = min(predict_bbox[0], result_bbox[0])
            ymin = min(predict_bbox[1], result_bbox[1])
            xmax = max(predict_bbox[2], result_bbox[2])
            ymax = max(predict_bbox[3], result_bbox[3])

            offsetWidth = (xmax-xmin)*1.6
            offsetHeight = (ymax-ymin)*1.6
            offsetLeft = util.get_rint(max(0, xmin-offsetWidth/2))
            offsetTop = util.get_rint(max(0, ymin-offsetHeight/2))
            offsetRight = util.get_rint(min(width, xmax+offsetWidth/2))
            offsetBottom = util.get_rint(min(height, ymax+offsetHeight/2))

            image = image[offsetTop:offsetBottom, offsetLeft:offsetRight, :]

            predict_bbox[0] -= offsetLeft
            predict_bbox[2] -= offsetLeft
            result_bbox[0] -= offsetLeft
            result_bbox[2] -= offsetLeft
            predict_bbox[1] -= offsetTop
            predict_bbox[3] -= offsetTop
            result_bbox[1] -= offsetTop
            result_bbox[3] -= offsetTop

            fig = plt.figure(figsize=(20, 20))

            plt.title(result_annotation['image_id'])

            plt.imshow(image)

            current_axis = plt.gca()

            current_axis.add_patch(plt.Rectangle(
                (predict_bbox[0], predict_bbox[1]), predict_bbox[2]-predict_bbox[0], predict_bbox[3]-predict_bbox[1], color=predict_color, fill=False, linewidth=3, linestyle='dashed'))
            current_axis.add_patch(plt.Rectangle(
                (result_bbox[0], result_bbox[1]), result_bbox[2]-result_bbox[0], result_bbox[3]-result_bbox[1], color=result_color, fill=False, linewidth=3))

            text = 'edge timer: ' + str(result_annotation['edge_timer']/10) + 's\ncorner timer: '+str(
                result_annotation['corner_timer']/10) + '\nIOU: ' + format(iou, '0.2f')

            plt.text(0, 0, text, ha='left', va='top',
                     fontdict={'color': judge_color, 'size': 30}, bbox={'edgecolor': judge_color, 'facecolor': 'white'})

            plt.show()
