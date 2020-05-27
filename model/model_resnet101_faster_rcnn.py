import torchvision
from torch import nn

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


class ModelResnet101FasterRCNN(FasterRCNN):
    def __init__(self, data_conf, model_conf):

        print("Creating model backbone with " + str(model_conf["hyperParameters"]["net"]))
        backbone_nn = torchvision.models.__dict__[model_conf["hyperParameters"]["net"]](pretrained=True)

        if model_conf["hyperParameters"]["freeze_pretrained_gradients"]:
            print("Using backbone as fixed feature extractor")
            modules = list(backbone_nn.children())[:-1]  # delete the last fc layer.
            backbone_nn = nn.Sequential(*modules)

            # FasterRCNN needs to know the number of
            # output channels in a backbone. For resnet101, it's 2048
            for param in backbone_nn.parameters():
                param.requires_grad = False
            backbone_nn.out_channels = model_conf["hyperParameters"]["net_out_channels"]
        else:
            print("Using fine-tuning of the model")
            modules = list(backbone_nn.children())[:-1]  # delete the last fc layer.
            backbone_nn = nn.Sequential(*modules)

            # FasterRCNN needs to know the number of
            # output channels in a backbone. For resnet101, it's 2048
            for param in backbone_nn.parameters():
                param.requires_grad = True
            backbone_nn.out_channels = model_conf["hyperParameters"]["net_out_channels"]
        #

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios

        anchor_ratios = tuple(model_conf["hyperParameters"]["anchor_ratios"])
        anchor_sizes = tuple(model_conf["hyperParameters"]["anchor_scales"])

        print("anchor_ratios = " + str(anchor_ratios))
        print("anchor_sizes = " + str(anchor_sizes))

        anchor_generator = AnchorGenerator(sizes=(anchor_sizes,),
                                           aspect_ratios=(anchor_ratios,))

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.

        rpn_pooling_size = model_conf["hyperParameters"]["pooling_size"]

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                        output_size=rpn_pooling_size,
                                                        sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        super().__init__(backbone=backbone_nn,
                         num_classes=len(data_conf["classes_available"]),
                         image_mean=model_conf["hyperParameters"]["normalization_mean"],
                         image_std=model_conf["hyperParameters"]["normalization_std"],
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler,
                         rpn_pre_nms_top_n_train=model_conf["hyperParameters"]["rpn_pre_nms_top_n_train"],
                         rpn_post_nms_top_n_train=model_conf["hyperParameters"]["rpn_post_nms_top_n_train"],
                         rpn_nms_thresh=model_conf["hyperParameters"]["rpn_nms_thresh"],
                         min_size=model_conf["hyperParameters"]["min_size_image"],
                         max_size=model_conf["hyperParameters"]["max_size_image"])
