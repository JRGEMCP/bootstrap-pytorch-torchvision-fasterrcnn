import os
import pickle
import pprint
import time

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data_loader.customcocodataset_debuggers import vis_detections, vis_ground_truth
from model.model_resnet101_faster_rcnn import ModelResnet101FasterRCNN
from model.model_resnet50_faster_rcnn import get_model_instance_segmentation
from references.detection import utils


class ModelScorer(object):
    """
    To be exposed via REST engine
    """
    def __init__(self, config=None):
        print("load your pkl and models here")
        self.scalar = pickle.load(open("models/scalar.pkl", "rb"))
        self.model = pickle.load(open("models/model.pkl", "rb"))

    def predict(self, data):
        data = self.scalar.transform([data])
        return self.model.prediction(data)

    def evaluate(self, x, y):
        x = self.scalar.transform(x)
        y_pred = self.model.predict(x)

        predictions = [round(value) for value in y_pred]
        # here you can do some accuracy or f_scores
        # accuracy = accuracy_score(y, predictions)
        accuracy = .09
        return {'accuracy': (accuracy * 100)}


@torch.no_grad()
def evaluate(data_conf, model_conf, **kwargs):

    start_time = time.time()

    print("Beginning evaluation of model")
    print("Using model_conf:")
    pprint.pprint(model_conf)

    if model_conf["pytorch_engine_scoring"]["enable_cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(1)

    #testing_dataset = CustomCocoDataset(data_conf=data_conf, model_conf=model_conf, testing_mode_on=True)

    testing_dataset = ImageFolder(root=data_conf["demo_in_image_dir"], transform=transforms.Compose([transforms.ToTensor()]))

    testing_data_loader = torch.utils.data.DataLoader(testing_dataset,
                                                       batch_size=model_conf["hyperParameters"]["batch_size"],
                                                       shuffle=False,
                                                       num_workers=model_conf["pytorch_engine_scoring"]["num_workers"],
                                                       collate_fn=utils.collate_fn)

    if model_conf["pytorch_engine"]["test_dataloader"]:

        for images, targets in testing_data_loader:

            images = list(img.to(device) for img in images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in targets]
            visualize_data(data_conf, model_conf, images, outputs)

        return

    if model_conf["hyperParameters"]["net"] == "spineless_model":
        model = get_model_instance_segmentation(data_conf=data_conf, model_conf=model_conf)
    else:
        model = ModelResnet101FasterRCNN(data_conf, model_conf)

    input_dir = data_conf["models_workspace_dir"] \
                + "/" \
                + model_conf["hyperParameters"]["net"] \
                + "/" \
                + data_conf["demo_model_dir"]

    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}.pth'.format(
                                 model_conf["hyperParameters"]["testing"]["check_session"],
                                 model_conf["hyperParameters"]["testing"]["check_epoch"]))

    print("load checkpoint %s" % load_name)
    model_checkpoint = torch.load(load_name)

    model.load_state_dict(model_checkpoint["model"])

    model.to(device)

    model.eval()

    for images, targets in testing_data_loader:
        try:
            images = list(img.to(device) for img in images)

            outputs = model(images)
            print("received outputs")
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

            image_ids = {}
            for filename, id in testing_data_loader.dataset.imgs:
                image_ids[id] = filename

            if model_conf["hyperParameters"]["testing"]["enable_visualization"]:

                visualize_data(data_conf=data_conf,
                               model_conf=model_conf,
                               images=images,
                               image_ids=image_ids,
                               metadatas=outputs)
        except Exception as excp:
            print("Exception occurred on an image set " + str(images) + ".. skipping")
            raise excp
    print("exited image loop")

    print("Evaluation complete")


def visualize_data(data_conf, model_conf, images, metadatas, image_ids=None):

    assert(len(images) == len(metadatas))

    for i in range(len(images)):
        binary_image = transforms.ToPILImage()(images[i]).convert("RGB")
        if model_conf["pytorch_engine"]["test_dataloader"]:

            vis_ground_truth(data_conf=data_conf,
                             model_conf=model_conf,
                             im=binary_image,
                             class_names=data_conf["classes_available"],
                             targets=metadatas[i])
        else:

            vis_detections(data_conf=data_conf,
                           model_conf=model_conf,
                           im=binary_image,
                           class_names=data_conf["classes_available"],
                           predictions=metadatas[i],
                           image_ids=image_ids)
    print("Results completed, check folder ")
    print(data_conf["demo_out_image_dir"] + "/" + model_conf["hyperParameters"]["net"] + "/" + data_conf["image_data_testing_id"])


if __name__ == "__main__":
    import json

    with open("../dataset.json") as f:
        config_json = json.load(f)

    with open("../config.json") as fp:
        model_conf = json.load(fp)

    evaluate(data_conf=config_json, model_conf=model_conf)
