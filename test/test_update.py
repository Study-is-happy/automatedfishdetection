import shutil
import numpy as np

import util
import config


results = {'correct': 0, 'adjust': 0, 'background': 0}


def random_bbox(bbox, image_width, image_height):
    width = bbox.xmax-bbox.xmin
    height = bbox.ymax-bbox.ymin
    bbox.xmin -= np.random.randn()*width/20
    bbox.xmax += np.random.randn()*width/20
    bbox.ymin -= np.random.randn()*height/20
    bbox.ymax += np.random.randn()*height/20

    util.norm_bbox(bbox, image_width, image_height)


for image_name in util.read_image_file(config.predict_image_file_path)[1:]:

    predict_annotation = util.read_annotation_file(
        config.predict_annotation_dir + image_name+'.xml')

    gt_annotation = util.read_annotation_file(config.processing_val_annotation_dir +
                                              image_name+'.xml')

    update_bboxes = []

    for predict_bbox in predict_annotation['bboxes']:

        for gt_bbox in gt_annotation['bboxes']:

            iou = util.get_bbox_iou(predict_bbox, gt_bbox)

            # if iou >= 0.9 \
            #         and predict_bbox.xmin >= gt_bbox.xmin \
            #         and predict_bbox.ymin >= gt_bbox.ymin \
            #         and predict_bbox.xmax >= gt_bbox.xmax \
            #         and predict_bbox.ymax >= gt_bbox.ymax:

            if iou >= 0.85:

                print(iou)

                predict_bbox.label = gt_bbox.label
                update_bboxes.append(predict_bbox)
                results['correct'] += 1
                break

            elif iou >= 0.3:

                print(iou)

                random_bbox(
                    gt_bbox, gt_annotation['width'], gt_annotation['height'])
                gt_bbox.adjust = 1
                update_bboxes.append(gt_bbox)
                results['adjust'] += 1
                break

        else:
            predict_bbox.label = 'background'
            update_bboxes.append(predict_bbox)
            results['background'] += 1

    shutil.copy(config.unannotated_dir+image_name+'.jpg',
                config.update_image_dir)

    util.write_image_name(
        config.update_image_file_path, image_name)

    util.write_xml(config.update_image_dir, config.update_annotation_dir, image_name,
                   gt_annotation['width'], gt_annotation['height'], update_bboxes)

print(results)
