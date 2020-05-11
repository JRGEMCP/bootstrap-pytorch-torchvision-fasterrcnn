import os
import pprint
import datetime
import time

import torch

from model.model_resnet101_faster_rcnn import ModelResnet101FasterRCNN
from data_loader.customcocodataset import CustomCocoDataset
from references.detection import utils
from model.model_resnet50_faster_rcnn import get_model_instance_segmentation
from references.detection.engine import train_one_epoch


def train(data_conf, model_conf, **kwargs):
    start_time = time.time()

    print("Using hyperParameters:")
    pprint.pprint(model_conf)

    output_dir = data_conf["models_workspace_dir"] + "/" + model_conf["hyperParameters"]["net"] + "/" \
                 + data_conf["image_data_training_id"] \
                 + "_" + str(datetime.datetime.now()).replace(" ", "-").replace(":", "_")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_conf["pytorch_engine"]["enable_cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(1)

    custom_dataset = CustomCocoDataset(data_conf=data_conf, model_conf=model_conf)

    training_data_loader = torch.utils.data.DataLoader(custom_dataset,
                                                       batch_size=model_conf["hyperParameters"]["batch_size"],
                                                       shuffle=True,
                                                       num_workers=model_conf["pytorch_engine"]["num_workers"],
                                                       collate_fn=utils.collate_fn)

    if model_conf["hyperParameters"]["net"] == "spineless_model":
        model = get_model_instance_segmentation(data_conf=data_conf, model_conf=model_conf)
    else:
        model = ModelResnet101FasterRCNN(data_conf, model_conf)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if model_conf["hyperParameters"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params=params,
                                     lr=model_conf["hyperParameters"]["learning_rate"],
                                     weight_decay=model_conf["hyperParameters"]["learning_weight_decay"])
    elif model_conf["hyperParameters"]["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params=params,
                                    lr=model_conf["hyperParameters"]["learning_rate"],
                                    momentum=model_conf["hyperParameters"]["momentum"],
                                    weight_decay=model_conf["hyperParameters"]["learning_weight_decay"])
    else:
        raise Exception("You must configure an optimizer within hyperParameters.  For example 'sgd'")

    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
    #                                               step_size=model_conf["hyperParameters"]["learning_decay_step"],
    #                                               gamma=model_conf["hyperParameters"]["learning_decay_gamma"])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=model_conf["hyperParameters"]["learning_decay_milestones"],
                                                        gamma=model_conf["hyperParameters"]["learning_decay_gamma"])

    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
    #                                                      gamma=model_conf["hyperParameters"]["learning_decay_gamma"])

    if model_conf["pytorch_engine"]["enable_tfb"]:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(output_dir + "/logs")
    else:
        logger = None

    for epoch in range(model_conf["hyperParameters"]["epoch_start"], model_conf["hyperParameters"]["epoch_max"] + 1):

        # train for one epoch, printing every 10 iterations
        train_one_epoch(model_conf=model_conf,
                        model=model,
                        optimizer=optimizer,
                        data_loader=training_data_loader,
                        device=device,
                        epoch=epoch,
                        tfb_logger=logger)

        # update the learning rate
        lr_scheduler.step()

        if model_conf["pytorch_engine"]["enable_tfb"]:
            logger.add_scalars(main_tag='logs_s_{}/lr'.format("1"),
                                   tag_scalar_dict={"lr": lr_scheduler.get_lr()},
                                   global_step=epoch)

        if epoch % 5 == 0 or epoch == model_conf["hyperParameters"]["epoch_max"]:
            save_name = os.path.join(output_dir,
                                     'faster_rcnn_{}_{}.pth'.format(model_conf["pytorch_engine"]["session"], epoch))
            torch.save({
                'session': model_conf["pytorch_engine"]["session"],
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'model_conf': model_conf,
                'data_conf': data_conf,
            }, save_name)
            print('save model: {}'.format(save_name))

    if model_conf["pytorch_engine"]["enable_tfb"]:
        logger.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import json

    with open("../dataset.json") as f:
        config_json = json.load(f)

    with open("../config.json") as fp:
        model_conf = json.load(fp)

    train(data_conf=config_json, model_conf=model_conf)
