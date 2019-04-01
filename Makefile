SHELL := /bin/bash

.PHONY : test1
test1 :
	-python3 demo2.py \
		--video_filepath=./data/original/video-camera01/01_02_1.avi \
		--output_dir=./output \
		--deep_sort_model_path=./model_data/mars-small128.pb \
		--yolo_model_path=./model_data/yolo.h5 \
		--yolo_anchors_path=./model_data/yolo_anchors.txt \
		--yolo_classes_path=./model_data/coco_classes.txt \
		--gpu=0 \
		--seed=0 \
		--log_len=1000 \
		--write_video_flag=True \
		--nn_metric=cosine \
		--nn_matching_threshold=0.3 \
		--nn_budget \
		--tracker_max_iou_distance=0.7 \
		--tracker_max_age=30 \
		--tracker_n_init=3 \
		--nms_max_overlap=1.0

.PHONY : webcam
webcam :
	python3 camera.py \
		--camera_num=0 \
		--deep_sort_model_path=./model_data/mars-small128.pb \
		--yolo_model_path=./model_data/yolo.h5 \
		--yolo_anchors_path=./model_data/yolo_anchors.txt \
		--yolo_classes_path=./model_data/coco_classes.txt \
		--gpu=0 \
		--seed=0 \
		--nn_metric=cosine \
		--nn_matching_threshold=0.3 \
		--nn_budget \
		--tracker_max_iou_distance=0.7 \
		--tracker_max_age=30 \
		--tracker_n_init=3 \
		--nms_max_overlap=1.0

LIST_NN_METRIC="cosine euclidean"
LIST_NN_MATCHING_THRESHOLD="0.1 0.3 0.5 0.7 1.0"
LIST_TRACKER_MAX_IOU_DISTANCE="0.1 0.3 0.5 0.7"
LIST_TRACKER_MAX_AGE="30 70 110 150 190 230 270 4100"
LIST_TRACKER_N_INIT="3 5 7"

tracker_max_age30 :
	LIST_TRACKER_MAX_AGE="30" GPU="0" ./demo2.bash.sh

tracker_max_age70 :
	LIST_TRACKER_MAX_AGE="70" GPU="1" ./demo2.bash.sh

tracker_max_age110 :
	LIST_TRACKER_MAX_AGE="110" GPU="2" ./demo2.bash.sh

tracker_max_age150 :
	LIST_TRACKER_MAX_AGE="150" GPU="3" ./demo2.bash.sh

tracker_max_age190 :
	LIST_TRACKER_MAX_AGE="190" GPU="4" ./demo2.bash.sh

tracker_max_age230 :
	LIST_TRACKER_MAX_AGE="230" GPU="5" ./demo2.bash.sh

tracker_max_age270 :
	LIST_TRACKER_MAX_AGE="270" GPU="6" ./demo2.bash.sh

tracker_max_age310 :
	LIST_TRACKER_MAX_AGE="310" GPU="7" ./demo2.bash.sh

#30 70 110 150 190 230 270 410

