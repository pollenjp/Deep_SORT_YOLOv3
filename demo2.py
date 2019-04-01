"""
@author pollenJP
@email polleninjp@gmail.com
"""

if __name__ == "__main__":
    import os
    import sys
    import pathlib
    import re
    import pprint
    import time
    import argparse
    import shutil

    import numpy as np
    import cv2
    print("cv2.__version__ : {}".format(cv2.__version__))
    import tensorflow as tf
    print("tensorflow.__version__ : {}".format(tf.__version__))

    import timeit
    import warnings
    from PIL import Image

    parser = argparse.ArgumentParser()
    #==============================
    parser.add_argument('--video_filepath', action='store', nargs=None,
                                const=None,
                                default="./data/original/video-camera01/01_02_1.avi",
                                type=str, choices=None, required=True,
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
    parser.add_argument('--log_len', action='store', nargs='?', const=None,
                                default=1000, type=int, choices=None,
                                required=False,
                                help="Progress Log. Divide total frame int `log_len`.")
    parser.add_argument('--write_video_flag', action='store', nargs='?',
                                const=None, default=True, type=bool,
                                choices=None, required=False,
                                help="write video or not")
    parser.add_argument('--output_dir', action='store', nargs='?',
                                const=None, default=None, type=str,
                                choices=None, required=True,
                                help="Save all of the frame results.")
    parser.add_argument('--refresh_flags', action='store_true', required=False,
                                help="Initialize output directory, if it already exists.")

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
                                    + "(ex: 1 => Not drop any bounding boxes.)" \
                                    + "(ex: 0.5 => スコアの高い`bbox1`の一部に," \
                                    + "自身の面積の半分(0.5)以上が含まれている" \
                                    + "`bbox2`が存在すれば`bbox2`をdropする.)")
    args = parser.parse_args()

    #===========================================================================
    # hyper parameter

    # nn_matching.NearestNeighborDistanceMetric
    nn_metric = args.nn_metric
    # (0.1, 0.3, 0.5, 0.7, 1.0)
    nn_matching_threshold = args.nn_matching_threshold
    nn_budget = args.nn_budget

    # Tracker
    # intersection of unit (0.1, 0.3, 0.5, 0.7)
    tracker_max_iou_distance = args.tracker_max_iou_distance
    # 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    tracker_max_age          = args.tracker_max_age
    tracker_n_init           = args.tracker_n_init

    # preprocessing.non_max_suppression
    # 0.3, 0.5, 0.7, 1.0
    nms_max_overlap = args.nms_max_overlap

    hyper_param_id = \
        "NN-metric{}-thresh{}-budget{}" \
        .format(nn_metric, nn_matching_threshold, nn_budget)
    hyper_param_id += \
        "__TRACKER-iou{}-age{}-ninit{}" \
        .format(tracker_max_iou_distance, tracker_max_age, tracker_n_init)
    hyper_param_id += "__overlap{}".format(nms_max_overlap)

    #===========================================================================
    # seed
    np.random.seed(seed=args.seed)
    tf.random.set_random_seed(seed=args.seed)
    # gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # path
    video_filepath = args.video_filepath
    yolo_model_path = args.yolo_model_path
    yolo_anchors_path = args.yolo_anchors_path
    yolo_classes_path = args.yolo_classes_path
    deep_sort_model_path = args.deep_sort_model_path
    check_list = [
        video_filepath,
        yolo_model_path,
        yolo_anchors_path,
        yolo_classes_path,
        deep_sort_model_path,
    ]
    for path in check_list:
        assert os.path.exists(path), "Not exists. : {}".format(path)
    # var
    # remove suffix from video filename
    video_id = os.path.basename(video_filepath)
    video_id = video_id[:-1-len(video_id.split(".")[-1])]

    refresh_flags = args.refresh_flags
    log_len = args.log_len
    write_video_flag = args.write_video_flag
    output_dir = args.output_dir

    # make a directory to save images
    if output_dir is not None:
        output_Path = pathlib.Path(os.path.realpath(output_dir))
        if not output_Path.exists():
            output_Path.mkdir(mode=0o755, parents=False, exist_ok=False)

        write_image_dir_Path = output_Path / video_id / hyper_param_id
        if write_image_dir_Path.exists():
            if refresh_flags is False:
                # confirm
                answer = ""
                while answer not in ["y", "n"]:
                    answer = input(
                        "{} already exists. Remove this directory [Y/N]? "
                        .format(write_image_dir_Path)).lower()
                assert answer == "y", "{} already exists.".format(write_image_dir_Path)
            # remove directory
            shutil.rmtree(path=str(write_image_dir_Path))
        write_image_dir_Path.mkdir(mode=0o755, parents=True, exist_ok=False)

        # `img/`
        (write_image_dir_Path / "img").mkdir(mode=0o755)

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
    tracker_instance = tracker.Tracker(metric           = metric,
                                       max_iou_distance = tracker_max_iou_distance,
                                       max_age          = tracker_max_age,
                                       n_init           = tracker_n_init)

    #===========================================================================
    # Get Video
    video_capture = cv2.VideoCapture(video_filepath)
    assert video_capture.isOpened(), "video capture failed. : {}".format(video_filepath)
    frame_total_num = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("frame_total_num : {}".format(frame_total_num))

    if write_video_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))

        if False:
            # AVI
            v_suffix = "avi"
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_output_fname = str(write_image_dir_Path / "{}-result.{}".format(video_id, v_suffix))
            video_output = cv2.VideoWriter(video_output_fname, fourcc, 15, (w, h))
        else:
            # MP4
            v_suffix = "mp4"
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            video_output_fname = str(write_image_dir_Path / "{}-result.{}".format(video_id, v_suffix))
            video_output = cv2.VideoWriter(video_output_fname, fourcc, 15, (w, h))

        list_file = open('detection.txt', 'w')
        frame_index = -1 

    # init
    fps = 0.0

    #===========================================================================
    # Load YOLO
    yolov3 = yolo.YOLO(model_path  = yolo_model_path,
                      anchors_path = yolo_anchors_path,
                      classes_path = yolo_classes_path)


    #===========================================================================
    # Log Preprocess
    if frame_total_num % log_len == 0:
        log_interval = frame_total_num // log_len
    else:
        log_interval = (frame_total_num + log_len) // log_len
    log_idxes = [i for i in range(log_interval-1, frame_total_num, log_interval)]
    print("len(log_idxes) : {}".format(len(log_idxes)))


    #===========================================================================
    # Loop
    start_time = time.time()
    for frame_counter in range(frame_total_num):
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            print("video_capture.read() failed.")
            print("frame_counter : {}/{}".format(frame_counter, frame_total_num))
            # なぜか`frame_total_num`までカウントする前にbreakしてしまう.
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
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        if write_video_flag:
            # save a frame
            video_output.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps + (1./(timeit.time.time()-t1))) / 2

        # save each frame result
        if output_dir is not None:
            filename = "{}-frame{}.jpeg".format(
                video_id,
                str(frame_counter).zfill(len(str(frame_total_num))))
            filepath = str(write_image_dir_Path / "img" / filename)
            cv2.imwrite(filepath, frame)

        if frame_counter == log_idxes[0]:
            elapsed_time = time.time() - start_time
            pct_progress = float(frame_counter / frame_total_num) * 100
            log_idxes.pop(0)
            print(
                "=== {:9.5f}% === | frame {:8}/{:8} | fps : {:8.4} | elapsed time : {:8.2}"
                .format(pct_progress, frame_counter, frame_total_num, fps, elapsed_time))

    # END Log
    elapsed_time = time.time() - start_time
    pct_progress = float(frame_counter / frame_total_num) * 100
    print(
        "=== {:9.5f}% === | frame {:8}/{:8}| fps : {:8.4} | elapsed time : {:8.2}"
        .format(pct_progress, frame_counter, frame_total_num, fps, elapsed_time))

    video_capture.release()
    if write_video_flag:
        video_output.release()
        list_file.close()
