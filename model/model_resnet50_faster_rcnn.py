import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model_instance_segmentation(data_conf, model_conf):
    num_classes = len(data_conf["classes_available"])
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    for param in model.parameters():
        param.requires_grad = True
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
