# import the necessary packages
import numpy as np
import sys
import time
import cv2
import os
import json
import base64
import io
import traceback
from PIL import Image
from flask import Flask, request, jsonify, Response

# construct the argument parse and parse the arguments
confthres = 0.3
nmsthres = 0.1

#initializing the flask app
app = Flask(__name__)

def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    lpath=os.path.sep.join([yolo_path, labels_path])

    print(yolo_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def do_prediction(image,net,LABELS, image_id):

    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    #print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    result = {} # Creating an empty dictionary to store the result.

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
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

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # TODO Prepare the output as required to the assignment specification
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        # Create a dictionary for the current detected object containing label, accuracy, and rectangle information
        for i in idxs.flatten():
            object_info = {"label": LABELS[classIDs[i]],"accuracy": confidences[i],
                "rectangle": {
                    "height": boxes[i][0],
                    "left": boxes[i][1],
                    "top": boxes[i][2],
                    "width": boxes[i][3]
            }}
            # Append the object_info dictionary to the 'objects' key in the result dictionary.
            # If the 'objects' key doesn't exist, create an empty list and append to it.
            if 'objects' not in result:
                result['objects'] = []
            result['objects'].append(object_info)

        result['id'] = image_id
        # Convert the result dictionary to a JSON string and return it
        return jsonify(result)
        


## argument
# if len(sys.argv) != 3:
#     raise ValueError("Argument list is wrong. Please use the following format:  {} {} {}".
#                      format("python iWebLens_server.py", "<yolo_config_folder>", "<Image file path>"))

# yolo_path  = str(sys.argv[1])




#TODO, you should  make this console script into webservice using Flask
# Defining the Flask API route for handling object detection requests
@app.route('/api/object_detection', methods=['POST'])
# main method
def main():
    try:
        # Get the JSON data from the request
        image_j = json.loads(request.json)
        # Extract the image's unique ID
        img_id = image_j['id']
        # Decode the base64 string containing the image
        img_base = base64.b64decode(image_j['image'])
        # Convert the decoded image data into an image object
        img_obj = Image.open(io.BytesIO(img_base))
        # Transform the image object into a NumPy array
        img_np = np.array(img_obj)
        # Change the image color space from BGR to RGB
        fin_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        # Load the YOLO object detection model
        nets = load_model(CFG, Weights)
        # Use the model to detect objects in the image
        response = do_prediction(fin_img, nets, Lables, img_id)
        # Return the detection results as a JSON response
        return response


    except Exception as e:
        # In case of an exception, return a message with the exception details
        return "Exception  {}".format(e)

    
@app.route("/")
def test():
    return "It is working!!!"

if __name__ == '__main__':
    ## Yolov3-tiny versrion
    yolo_path  = "yolo_tiny_configs"
    labelsPath= "coco.names"
    cfgpath= "yolov3-tiny.cfg"
    wpath= "yolov3-tiny.weights"

    Lables=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)

    app.run(debug=True, threaded=True, host="0.0.0.0",port=5000)


