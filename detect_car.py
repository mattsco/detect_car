import argparse
import time
import scipy.misc
import os
import cv2
import imageio
import json
import numpy as np
from collections import Counter


def detect_car(yolo_model_path, images_path, output_path, objects_path):
    args = {'threshold': 0.05, 'yolo': yolo_model_path, 'confidence': 0.05}


    LABELS = ["car"]

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = [[41,177,171]]

    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


    for filename in os.listdir(images_path):
        c0 = Counter()
        c0["image"] = filename
        if filename.endswith(".png"):# and filename.startswith("snap_2019"):
            args["image"] = os.path.join(images_path, filename)
            print filename

            # load our input image and grab its spatial dimensions
            #image = cv2.imread(args["image"])
            try:
                image = imageio.imread(args["image"])
                (H, W) = image.shape[:2]

                # determine only the *output* layer names that we need from YOLO
                ln = net.getLayerNames()
                ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


                # construct a blob from the input image and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes and
                # associated probabilities
                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln)
                end = time.time()

                # show timing information on YOLO
                print("[INFO] YOLO took {:.6f} seconds".format(end - start))

                # initialize our lists of detected bounding boxes, confidences, and
                # class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability) of
                        # the current object detection
                        scores = detection[5:]
                        #classID = np.argmax(scores)

                        classID = [scores[2]]
                        confidence = scores[2]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > args["confidence"]:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes' width and height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)


                # apply non-maxima suppression to suppress weak, overlapping bounding
                # boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

                # ensure at least one detection exists

                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the image
                        #if classIDs[i]==2:
                        color = COLORS[0]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format("car", confidences[i])
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, color, 1)
                        c0["car"]+=1


                if len(c0)>=1:
                    #scipy.misc.imsave(output_path+'/'+filename, image)
                    imageio.imwrite(output_path+'/'+filename, image)

                    object_path = objects_path+ "/"+ filename.split(".")[0]+".json"
                    with open(object_path, 'wt') as out:
                        c0 = dict(c0)
                        res = json.dumps(c0, sort_keys=True, indent=4, separators=(',', ': '))
                        out.write(res)

            except ValueError:
                pass