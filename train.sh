DATASDATASET_PATH='C:\Users\Hailin\OneDrive - The University of Liverpool\SURF23-24\Computer Vision-Based Traffic Accident Detection\datasets\hilichurl-detector.v1i.yolov8.yaml'

yolo yolov8 task=detect \
mode=mode = train \
modelmodel = yolov8n.pt \
data=data = $DATASET_PATH \
epochepochs = 10