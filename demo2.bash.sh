#!/bin/bash

#GPU=0
LIST_NN_METRIC="cosine euclidean"
LIST_NN_MATCHING_THRESHOLD="0.1 0.3 0.5 0.7 1.0"
LIST_TRACKER_MAX_IOU_DISTANCE="0.1 0.3 0.5 0.7"
#LIST_TRACKER_MAX_AGE="30 70 110 150 190 230 270 310"
LIST_TRACKER_N_INIT="3 5 7"

for NN_METRIC in ${LIST_NN_METRIC}; do
    for TRACKER_N_INIT in ${LIST_TRACKER_N_INIT}; do
        for TRACKER_MAX_IOU_DISTANCE in ${LIST_TRACKER_MAX_IOU_DISTANCE}; do
            for TRACKER_MAX_AGE in ${LIST_TRACKER_MAX_AGE}; do
                for NN_MATCHING_THRESHOLD in ${LIST_NN_MATCHING_THRESHOLD}; do
                    python3 demo2.py \
                        --video_filepath=./data/original/video-camera01/01_02_1.avi \
                        --output_dir=./output \
                        --deep_sort_model_path=./model_data/mars-small128.pb \
                        --yolo_model_path=./model_data/yolo.h5 \
                        --yolo_anchors_path=./model_data/yolo_anchors.txt \
                        --yolo_classes_path=./model_data/coco_classes.txt \
                        --gpu=${GPU} \
                        --seed=0 \
                        --log_len=1000 \
                        --write_video_flag=True \
                        --nn_metric=${NN_METRIC} \
                        --nn_matching_threshold=${NN_MATCHING_THRESHOLD} \
                        --nn_budget \
                        --tracker_max_iou_distance=${TRACKER_MAX_IOU_DISTANCE} \
                        --tracker_max_age=${TRACKER_MAX_AGE} \
                        --tracker_n_init=${TRACKER_N_INIT} \
                        --nms_max_overlap=1.0
                done
            done
        done
    done
done
