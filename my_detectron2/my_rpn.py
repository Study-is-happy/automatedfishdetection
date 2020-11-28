from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn_outputs import RPNOutputs, find_top_rpn_proposals
from my_detectron2.my_rpn_outputs import my_RPNOutputs, my_find_top_rpn_proposals

from detectron2.modeling.proposal_generator.rpn import build_rpn_head


@PROPOSAL_GENERATOR_REGISTRY.register()
class my_RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh = cfg.MODEL.RPN.NMS_THRESH
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight = cfg.MODEL.RPN.LOSS_WEIGHT
        self.ignore_prob = cfg.CUSTOM_IGNORE_PROB
        self.cls_loss_factor = cfg.CUSTOM_CLS_LOSS_FACTOR
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(
            cfg, [input_shape[f] for f in self.in_features])

    def forward(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances] or None
            loss: dict[Tensor]
        """
        gt_boxes = [
            x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        gt_classes = [
            x.gt_classes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        anchors = self.anchor_generator(features)
        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).

        if self.training:
            outputs = my_RPNOutputs(
                self.box2box_transform,
                self.anchor_matcher,
                self.num_classes,
                self.batch_size_per_image,
                self.positive_fraction,
                images,
                pred_objectness_logits,
                pred_anchor_deltas,
                anchors,
                self.boundary_threshold,
                gt_classes,
                gt_boxes,
                self.smooth_l1_beta,
                self.ignore_prob,
                self.cls_loss_factor
            )
            losses = {k: v * self.loss_weight for k,
                      v in outputs.losses().items()}
        else:
            outputs = RPNOutputs(
                self.box2box_transform,
                self.anchor_matcher,
                self.batch_size_per_image,
                self.positive_fraction,
                images,
                pred_objectness_logits,
                pred_anchor_deltas,
                anchors,
                self.boundary_threshold,
                gt_boxes,
                self.smooth_l1_beta,
            )
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            if self.training:
                proposals = my_find_top_rpn_proposals(
                    outputs.predict_proposals(),
                    outputs.predict_objectness_logits(),
                    outputs.gt_objectness_logits,
                    images,
                    self.nms_thresh,
                    self.pre_nms_topk[self.training],
                    self.post_nms_topk[self.training],
                    self.min_box_side_len,
                    self.training,
                )
            else:
                proposals = find_top_rpn_proposals(
                    outputs.predict_proposals(),
                    outputs.predict_objectness_logits(),
                    images,
                    self.nms_thresh,
                    self.pre_nms_topk[self.training],
                    self.post_nms_topk[self.training],
                    self.min_box_side_len,
                    self.training,
                )
            # For RPN-only models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            # For end-to-end models, the RPN proposals are an intermediate state
            # and this sorting is actually not needed. But the cost is negligible.

            inds = [p.objectness_logits.sort(descending=True)[
                1] for p in proposals]
            proposals = [p[ind] for p, ind in zip(proposals, inds)]

        return proposals, losses
