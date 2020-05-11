import os
import pickle
import pprint
import time

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data_loader.customcocodataset_debuggers import vis_detections
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

    # testing_dataset = CustomCocoDataset(data_conf=data_conf, model_conf=model_conf, testing_mode_on=True)

    testing_dataset = ImageFolder(root=data_conf["demo_in_image_dir"], transform=transforms.Compose([transforms.ToTensor()]))

    # print("found " + str(testing_dataset.get_num_categories()) + " categories in data")

    testing_data_loader = torch.utils.data.DataLoader(testing_dataset,
                                                       batch_size=model_conf["hyperParameters"]["batch_size"],
                                                       shuffle=False,
                                                       num_workers=model_conf["pytorch_engine_scoring"]["num_workers"],
                                                       collate_fn=utils.collate_fn)

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

    result = exec_evaluate(data_conf=data_conf,
                           model_conf=model_conf,
                           model=model, data_loader=testing_data_loader, device=device)

    print("Evaluation complete")


def visualize_result(data_conf, model_conf, images, image_ids, predictions):

    assert(len(images) == len(predictions))

    for i in range(len(images)):
        print("image tensor size is :" + str(images[i].size()))
        binary_image = transforms.ToPILImage()(images[i]).convert("RGB")
        tagged_binary_image = vis_detections(model_conf, binary_image, data_conf["classes_available"], predictions[i])
        # output tagged image to disk

        tagged_binary_image.save(data_conf["demo_out_image_dir"]
                                 + "/" +
                                 model_conf["hyperParameters"]["net"]
                                 + "/" +
                                 data_conf["image_data_testing_id"] + "/" +
                                 str(image_ids) + "_result.jpg", "JPEG")
        print("done with tagged image")


@torch.no_grad()
def exec_evaluate(data_conf, model_conf, model, data_loader, device):
    print("beginning actual eval")
    model.eval()

    print("entering image target loop")
    for images, targets in data_loader:
        try:
            images = list(img.to(device) for img in images)

            outputs = model(images)
            print("received outputs")
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

            if model_conf["hyperParameters"]["testing"]["enable_visualization"]:

                visualize_result(data_conf=data_conf,
                                 model_conf=model_conf,
                                 images=images,
                                 image_ids=targets[0]["image_id"],
                                 predictions=outputs)
        except Exception as excp:
            print("Exception occurred on an image set " + str(images) + ".. skipping")
            raise excp
    print("exited image loop")


if __name__ == "__main__":
    import json

    with open("../dataset_template.json") as f:
        config_json = json.load(f)

    with open("../config.json") as fp:
        model_conf = json.load(fp)

    evaluate(data_conf=config_json, model_conf=model_conf)
