from PIL import ImageDraw


def vis_detections(model_conf, im, class_names, predictions):

    print(str(predictions))
    """Visual debugging of detections."""
    for item in range(len(predictions)):
        bboxes = predictions["boxes"]
        labels = predictions["labels"]
        scores = predictions["scores"]

        assert len(bboxes) == len(labels) == len(scores)

        index = 0
        for box in bboxes.numpy():
            print("assessing score " + str(scores[index]))
            if scores[index] > model_conf["hyperParameters"]["testing"]["visualization_thresh"]:
                ImageDraw.ImageDraw(im=im).rectangle(xy=box, outline="yellow", width=1)
                ImageDraw.ImageDraw(im=im).text(xy=[box[0], box[1] + 15],
                                                text=class_names[index] + "@" + str(scores[index]),
                                                fill="blue")
                index += 1

        im.show()
    return im