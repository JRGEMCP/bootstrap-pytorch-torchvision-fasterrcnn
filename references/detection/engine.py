import math
import sys
import torch
import torchvision.models.detection.mask_rcnn
import references.detection.utils as utils


def train_one_epoch(model_conf, model, optimizer, data_loader, device, epoch, tfb_logger):
    print_freq = model_conf["hyperParameters"]["display_interval"]
    iterations_per_epoch = len(data_loader) / model_conf["hyperParameters"]["batch_size"]

    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    iterations = 0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        iterations += 1
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if tfb_logger is not None:
            info = {
                'loss': losses_reduced,
                'loss_box_reg': loss_dict["loss_box_reg"],
                'loss_classifier': loss_dict["loss_classifier"],
                'loss_objectness': loss_dict["loss_objectness"],
                'loss_rpn_box_reg': loss_dict["loss_rpn_box_reg"]
            }

            tfb_logger.add_scalars(main_tag='logs_s_{}/losses'.format("1"),
                                   tag_scalar_dict=info,
                                   global_step=(epoch * len(data_loader)) + iterations)

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types



