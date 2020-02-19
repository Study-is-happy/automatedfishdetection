from detectron2.data import DatasetCatalog
import numpy as np
from collections import OrderedDict

import config
import util


class Evaluator():

    def __init__(self, dataset_name):

        self.gt_instances = DatasetCatalog.get(dataset_name)
        self.reset()

    def reset(self):
        self.gt_index = 0
        self.npos_list = []
        self.tp_list = []
        self.confidence_list = []
        for _ in range(config.num_categories):
            self.npos_list.append(0)
            self.tp_list.append([])
            self.confidence_list.append([])

    def process(self, inputs, outputs):

        for output in outputs:

            gt_annotations = []

            for _ in range(config.num_categories):
                gt_annotations.append([])

            for gt_annotation in self.gt_instances[self.gt_index]['annotations']:

                category_id = gt_annotation['category_id']

                gt_annotation['detected'] = False

                gt_annotations[category_id].append(
                    gt_annotation)

                self.npos_list[category_id] += 1

            self.gt_index += 1

            output_instances = output['instances']
            fields = output_instances.get_fields()

            for category_id, score, bbox in zip(fields['pred_classes'], fields['scores'], fields['pred_boxes']):

                category_id = category_id.item()
                score = score.item()
                bbox = bbox.tolist()

                self.confidence_list[category_id].append(score)

                best_iou = 0.5
                best_gt_annotation = None

                for index, gt_annotation in enumerate(gt_annotations[category_id]):
                    iou = util.get_bboxes_iou(bbox, gt_annotation['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_annotation = gt_annotation

                if best_gt_annotation == None or best_gt_annotation['detected']:
                    self.tp_list[category_id].append(0)
                else:
                    best_gt_annotation['detected'] = True
                    self.tp_list[category_id].append(1)

    def evaluate(self):

        ap_list = []

        for index in range(config.num_categories):
            tp = np.array(self.tp_list[index])
            confidence = np.array(self.confidence_list[index])

            sorted_confidence_index = np.argsort(-confidence)
            tp = np.cumsum(tp[sorted_confidence_index])

            recall = tp/self.npos_list[index]
            precision = tp/(np.arange(len(tp))+1)

            mrecall = np.concatenate(([0.0], recall, [1.0]))
            mprecision = np.concatenate(([0.0], precision, [0.0]))

            for i in range(mprecision.size - 1, 0, -1):
                mprecision[i -
                           1] = np.maximum(mprecision[i - 1], mprecision[i])

            i = np.where(mrecall[1:] != mrecall[:-1])[0]
            ap = np.sum((mrecall[i + 1] - mrecall[i]) * mprecision[i + 1])
            ap_list.append(ap)

        mAP = np.mean(ap_list)
        result = OrderedDict()
        result['mAP'] = {'50': mAP}

        return result
