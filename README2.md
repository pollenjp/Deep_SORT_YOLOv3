

```
wget https://pjreddie.com/media/files/yolov3.weights
python3 convert.py yolov3.cfg yolov3.weights yolo.h5 -p
```


## `demo2.py`

```
python3 demo2.py \
    --video_filepath=./data/original/video-camera01/01_01_1.avi \
    --output_dir=./output \
    --deep_sort_model_path=./model_data/mars-small128.pb \
    --yolo_model_path=./model_data/yolo.h5 \
    --yolo_anchors_path=./model_data/yolo_anchors.txt \
    --yolo_classes_path=./model_data/coco_classes.txt \
    --gpu=0 \
    --seed=0 \
    --log_len=1000 \
    --write_video_flag=True \
    --max_cosine_distance=0.3 \
    --nn_budget \
    --nms_max_overlap=1.0
```

