## YOLO Based object Tracking

Create conda environemnt and install necessary packages
```
conda create -n yolo python=3.9
conda activate yolo
```
Install necessary packages:

```
pip install opencv-python
pip install matplotlib
```


Get YOLO weights from [YOLO](https://pjreddie.com/darknet/yolo/) webpage

```
wget https://pjreddie.com/media/files/yolov3.weights
```

Then go and download [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
and [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
```