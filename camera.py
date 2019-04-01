"""
@author pollenJP
@email polleninjp@gmail.com
"""
import colorsys

import cv2
print("cv2.__version__ : {}".format(cv2.__version__))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    #==============================
    parser.add_argument('--camera_num', action='store', nargs=None,
                                const=None, default=0,
                                type=int, choices=None, required=True,
                                help="yolov3 h5 model path")
    parser.add_argument('--deep_sort_model_path', action='store', nargs=None,
                                const=None,
                                default="./model_data/mars-small128.pb",
                                type=str, choices=None, required=True,
                                help="Deep SORT `.pb` model path")
    parser.add_argument('--yolo_model_path', action='store', nargs=None,
                                const=None, default="./model_data/yolo.h5",
                                type=str, choices=None, required=True,
                                help="yolov3 `.h5` model path")
    parser.add_argument('--yolo_anchors_path', action='store', nargs=None,
                                const=None,
                                default="./model_data/yolo_anchors.txt",
                                type=str, choices=None, required=True,
                                help="yolov3 anchors.txt")
    parser.add_argument('--yolo_classes_path', action='store', nargs=None,
                                const=None,
                                default="./model_data/coco_classes.txt",
                                type=str, choices=None, required=True,
                                help="yolov3 anchors.txt")

    #===========================================================================
    parser.add_argument('--gpu', action='store', nargs='?', const="", 
                                default="", type=str, choices=None,
                                required=False, help="GPU '0,1,...'")
    parser.add_argument('--seed', action='store', nargs='?', const=None,
                                default=None, type=int, choices=None, 
                                required=False, help='random seed')

    #===========================================================================
    # Definition of the parameters

    # nn_matching.NearestNeighborDistanceMetric
    parser.add_argument('--nn_metric', action='store', nargs='?',
                                const=None, default="cosine", type=str,
                                choices=["euclidean", "cosine"],
                                required=False, help="")
    parser.add_argument('--nn_matching_threshold', action='store', nargs='?',
                                const=None, default=0.3, type=float,
                                choices=None, required=False, help="")
    parser.add_argument('--nn_budget', action='store', nargs='?',
                                const=None, default=None, type=int,
                                choices=None, required=False, help="")

    # Tracker
    parser.add_argument('--tracker_max_iou_distance', action='store', nargs='?',
                                const=None, default=0.7, type=float,
                                choices=None, required=False, help="")
    parser.add_argument('--tracker_max_age', action='store', nargs='?',
                                const=None, default=30, type=int,
                                choices=None, required=False, help="")
    parser.add_argument('--tracker_n_init', action='store', nargs='?',
                                const=None, default=3, type=int,
                                choices=None, required=False, help="")

    # preprocessing.non_max_suppression
    # bboxの重なり度合いから選別
    parser.add_argument('--nms_max_overlap', action='store', nargs='?',
                                const=None, default=1.0, type=float,
                                choices=None, required=False,
                                help="nms_max_overlap range : (0, 1]." \
                                    +"(ex: 1 => Not drop any bounding boxes.)"\
                                    +"(ex: 0.5 => スコアの高い`bbox1`の一部に,"\
                                    +"自身の面積の半分(0.5)以上が含まれている"\
                                    +"`bbox2`が存在すれば`bbox2`をdropする.)")
    return parser.parse_args()

#===============================================================================
# main
#===============================================================================
def main(nn_metric,
         nn_matching_threshold,
         nn_budget,
         tracker_max_iou_distance,
         tracker_max_age,
         tracker_n_init,
         nms_max_overlap,
         camera_num,
         yolo_model_path,
         yolo_anchors_path,
         yolo_classes_path,
         deep_sort_model_path,
         seed=None, gpu=""):
    """
    Parameters
    ----------
    nn_metric : str
        "cosine" or "euclidean"
        Use in nn_matching.NearestNeighborDistanceMetric
    nn_matching_threshold : float
    nn_budget : int
    tracker_max_iou_distance : int
        In a final matching stage, we run intersection over union
        association as proposed in the original SORT algorithm
        on the set of unconfirmed and unmatched tracks of age n = 1
    tracker_max_age : int
        Tracks are terminated if they are not detected for `tracker_max_age` frames.
    tracker_n_init : int
        A new tracker undergoes a probationary period where the target needs to
        be eassociated with detecions to accumulate enough evidence
        in order to prevent tracking of false positive.
    nms_max_overlap : float, range 0 ~ 1
        preprocessing.non_max_suppression
    camera_num : int
        webcam device number
    yolo_model_path : str
        .h5 weight file
    yolo_anchors_path : str
        anchors file path
    yolo_classes_path : str
        classes filepath
    deep_sort_model_path : str
        weight filepath
    seed : int or None
        random seed
    gpu : str
        like "0,1"
    """
    import os
    import sys
    import pathlib
    import re
    import pprint
    import time
    import shutil

    import numpy as np
    import tensorflow as tf
    print("tensorflow.__version__ : {}".format(tf.__version__))

    import timeit
    import warnings
    from PIL import Image

    hyper_param_id = \
        "NN-metric-{}-thresh{}-budget{}" \
        .format(nn_metric, nn_matching_threshold, nn_budget)
    hyper_param_id += \
        "__TRACKER-iou{}-age{}-ninit{}" \
        .format(tracker_max_iou_distance, tracker_max_age, tracker_n_init)
    hyper_param_id += "__overlap{}".format(nms_max_overlap)

    # seed
    np.random.seed(seed=seed)
    tf.random.set_random_seed(seed=seed)
    # gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    # path
    check_list = [
        yolo_model_path,
        yolo_anchors_path,
        yolo_classes_path,
        deep_sort_model_path,
    ]
    for path in check_list:
        assert os.path.exists(path), "Not exists. : {}".format(path)

    #===========================================================================
    import yolo
    from deep_sort import preprocessing
    from deep_sort import nn_matching
    from deep_sort import detection
    from deep_sort import tracker
    from tools import generate_detections
    warnings.filterwarnings('ignore')

    #===========================================================================
    # deep_sort 
    encoder = \
        generate_detections.create_box_encoder(
            model_filename = deep_sort_model_path,
            input_name     = 'images',
            output_name    = 'features',
            batch_size     = 1)
    metric = \
        nn_matching.NearestNeighborDistanceMetric(
            metric             = nn_metric,
            matching_threshold = nn_matching_threshold,
            budget             = nn_budget)
    tracker_instance = \
        tracker.Tracker(metric           = metric,
                        max_iou_distance = tracker_max_iou_distance,
                        max_age          = tracker_max_age,
                        n_init           = tracker_n_init)

    # Get Video
    cap = cv2.VideoCapture(camera_num)
    assert cap.isOpened(), "webcam capture failed. : {}".format(camera_num)

    # init
    fps = 0.0

    # Load YOLO
    yolov3 = yolo.YOLO(model_path   = yolo_model_path,
                       anchors_path = yolo_anchors_path,
                       classes_path = yolo_classes_path)

    # Loop
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("webcam cap.read() failed.")
            break
        t1 = timeit.time.time()

        image = Image.fromarray(frame)
        # x,y,w,h
        boxs = yolov3.detect_image(image)
        # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [
            detection.Detection(tlwh=bbox, confidence=1.0, feature=feature)
            for bbox, feature in zip(boxs, features)
        ]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker_instance.predict()
        tracker_instance.update(detections)
        for track in tracker_instance.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()

            # bgr
            color = (0,0,255)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, 2)
            #cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            top_left = (int(bbox[0]), int(bbox[1]))

            font_face = cv2.FONT_HERSHEY_PLAIN
            label = str(track.track_id)
            text_size, _ = \
                cv2.getTextSize(text=label,
                                fontFace=font_face,
                                fontScale=2,
                                thickness=2)
            #top_left = (top_left[0], top_left[1] + text_size[1] - 1)
            bottom_right = (top_left[0] + text_size[0] - 1,
                            top_left[1] + text_size[1] - 1)
            bottom_left  = (top_left[0], bottom_right[1])
            cv2.rectangle(img=frame,
                          pt1=top_left,
                          pt2=bottom_right,
                          color=color,
                          thickness=-1)
            cv2.putText(img=frame,
                        text=label,
                        org=bottom_left,
                        fontFace=font_face,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=1)
 

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        fps = (fps + (1./(timeit.time.time()-t1))) / 2

        # show each frame result
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        # print log
        elapsed_time = time.time() - start_time
        print("fps : {:8.4} | elapsed time : {:8.2}".format(fps, elapsed_time))

    # END Log
    elapsed_time = time.time() - start_time
    print("fps : {:8.4} | elapsed time : {:8.2}".format(fps, elapsed_time))

    cap.release()



if __name__ == "__main__":

    args = parse_args()

    main(nn_metric = args.nn_metric,
         nn_matching_threshold = args.nn_matching_threshold,
         nn_budget = args.nn_budget,
         tracker_max_iou_distance = args.tracker_max_iou_distance,
         tracker_max_age = args.tracker_max_age,
         tracker_n_init = args.tracker_n_init,
         nms_max_overlap = args.nms_max_overlap,

         camera_num = args.camera_num,

         yolo_model_path = args.yolo_model_path,
         yolo_anchors_path = args.yolo_anchors_path,
         yolo_classes_path = args.yolo_classes_path,
         deep_sort_model_path = args.deep_sort_model_path,
         seed = args.seed,
         gpu = args.gpu)
