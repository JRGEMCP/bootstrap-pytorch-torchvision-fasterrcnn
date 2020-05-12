import os

from PIL import ImageDraw


def vis_detections(data_conf, model_conf, im, class_names, predictions, image_ids):

    print(str(predictions))
    """Visual debugging of detections."""
    for item in range(len(predictions)):
        bboxes = predictions["boxes"]
        labels = predictions["labels"]
        #image_ids = predictions["image_id"]
        scores = predictions["scores"]

        assert len(bboxes) == len(labels) == len(scores)

        index = 0
        for box in bboxes.numpy():
            print("assessing score " + str(scores[index]))
            if scores[index] > model_conf["hyperParameters"]["testing"]["visualization_thresh"]:
                target_label = str(class_names[labels.item()])
                print("We found an object, labeled: " + target_label)
                ImageDraw.ImageDraw(im=im).rectangle(xy=box, outline="yellow", width=1)
                ImageDraw.ImageDraw(im=im).text(xy=[box[0], box[1] + 15],
                                                text=target_label + "@" + str(scores[index]),
                                                fill="blue")
                index += 1

        #im.show()

        os.makedirs(data_conf["demo_out_image_dir"]
                                 + "/" +
                                 model_conf["hyperParameters"]["net"]
                                 + "/" +
                                 data_conf["image_data_testing_id"], exist_ok=True)

        # TODO: unhardcode from batch_size 1, ie index 0th image_id everytime

        path_tokens = image_ids[0].split("/")

        file_name = path_tokens[len(path_tokens) - 1]

        result_file_path = data_conf["demo_out_image_dir"] + "/" + model_conf["hyperParameters"]["net"] + "/" + data_conf["image_data_testing_id"] + "/" + file_name + "_result.jpg"

        if os.path.exists(result_file_path):
            os.remove(result_file_path)

        im.save(result_file_path, "JPEG")
    return im


def vis_ground_truth(data_conf, model_conf, im, class_names, targets):

    """Visual debugging of detections."""
    for item in range(len(targets)):
        bboxes = targets["boxes"]
        labels = targets["labels"]
        image_ids = targets["image_id"]

        assert len(bboxes) == len(labels)

        index = 0
        for box in bboxes.numpy():

            ImageDraw.ImageDraw(im=im).rectangle(xy=box, outline="yellow", width=1)
            ImageDraw.ImageDraw(im=im).text(xy=[box[0], box[1] + 15],
                                            text=class_names[labels.item()],
                                            fill="blue")
            index += 1

        #im.show()

        os.makedirs(data_conf["demo_out_image_dir"]
                                 + "/" +
                                 model_conf["hyperParameters"]["net"]
                                 + "/" +
                                 data_conf["image_data_testing_id"], exist_ok=True)

        # output tagged image to disk
        im.save(data_conf["demo_out_image_dir"]
                                 + "/" +
                                 model_conf["hyperParameters"]["net"]
                                 + "/" +
                                 data_conf["image_data_testing_id"] + "/" +
                                 str(image_ids.item()) + "_result.jpg", "JPEG")
    return im