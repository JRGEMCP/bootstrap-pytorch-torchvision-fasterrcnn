import os

import torch
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


class CustomCocoDataset(CocoDetection):

    def __init__(self, data_conf, model_conf):

        target_data_set = data_conf["image_data_training_id"]
        coco_filename = data_conf["coco_annotations_training_id"]

        self.coco_data_root = os.path.join(data_conf["image_pool_path"],
                                           target_data_set + "/" +
                                           data_conf["image_data_sub_dir"])

        self.coco_annotations_file = os.path.join(data_conf["image_pool_path"],
                                                  target_data_set,
                                                  "annotations",
                                                  coco_filename + ".json")

        super(CustomCocoDataset, self).__init__(root=self.coco_data_root,
                                                annFile=self.coco_annotations_file)

        self.categories = data_conf["classes_available"]

        print("found " + str(self.get_num_categories()) + " categories in data at: " + str(self.coco_data_root))

    def get_num_categories(self):
        return len(self.categories)

    def __getitem__(self, item):

        (pil_image, targets) = super(CustomCocoDataset, self).__getitem__(item)

        # get bounding box coordinates for each mask
        num_targets = len(targets)
        boxes = []
        for i in range(num_targets):
            box = targets[i]["bbox"]
            xmin = box[0]
            xmax = box[0] + box[2]
            ymin = box[1]
            ymax = box[1] + box[3]
            assert xmin < xmax and ymin < ymax
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_targets,), dtype=torch.int64)
        image_id = torch.tensor([item])
        areas = []
        for i in range(num_targets):
            areas.append(targets[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_targets,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        image = F.to_tensor(pil_image)

        return image, target


if __name__ == "__main__":
    import json

    with open("../../dataset_template.json") as f:
        config_json = json.load(f)

    with open("../../config.json") as fp:
        model_conf = json.load(fp)

    loader = CustomCocoDataset(data_conf=config_json, model_conf=model_conf)

    print(loader.__len__())
    print(loader.__getitem__(4))

    print("Done")
