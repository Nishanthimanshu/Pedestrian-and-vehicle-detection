import cv2
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow_datasets.core import dataset_info
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import numpy as np


import untitled
from untitled import DecodePredictions, RetinaNetLoss

loss_fn = RetinaNetLoss(8)
model = keras.models.load_model("Retinanet_model", compile=False)
optimizer = tf.optimizers.SGD(learning_rate=untitled.learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

int2str = { 0: "Car", 1: "Van", 2: "Truck", 3: "Pedestrian", 4: "Person_sitting", 5: "Cyclist", 6: "Tram", 7: "Misc",
           }
COLORS = np.random.uniform(0, 255, size=(len(int2str), 3))


def prepare_image(image):
    image, _, ratio = untitled.resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture("Chicago_360p.mp4")

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, frame = self.video.read()
        image = tf.cast(frame, dtype=tf.float32)
        input_image, ratio = prepare_image(image)
        detections = inference_model.predict(input_image)
        num_detections = detections.valid_detections[0]
        class_names = [int2str[int(x)] for x in detections.nmsed_classes[0][:num_detections]]
        boxes = detections.nmsed_boxes[0][:num_detections] / ratio
        classes = class_names
        scores = detections.nmsed_scores[0][:num_detections]

        for box, _cls, score in zip(boxes, classes, scores):
            text = "{}: {:.2f}".format(_cls, score)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(frame, (x1, y1), (x2, y2),color=COLORS[classes.index(_cls)])
            y = y1 - 15 if y1 - 15 > 15 else y1 + 15
            cv2.putText(frame, text, (x1, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[classes.index(_cls)], 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
