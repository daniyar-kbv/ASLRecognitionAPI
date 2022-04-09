import tensorflow as tf
import numpy as np
import cv2


def detect_fn(image, detection_model):
    """
    Detect objects in image.

    Args:
      image: (tf.tensor): 4D input image

    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def infer(image_np, detection_model, category_index):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    index = np.argmax([score for score in detections['detection_scores'] if score < 0.9])

    detection_score = detections['detection_scores'][index]
    detection_box = detections['detection_boxes'][index]
    detection_class = detections['detection_classes'][index] + 1
    detection_class_name = category_index[detection_class]['name']

    return detection_class_name, detection_box, detection_score
