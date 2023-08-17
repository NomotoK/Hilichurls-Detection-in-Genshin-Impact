import torch
import math
from ultralytics import YOLO
from PIL import Image
import cv2


print("torch version: " + torch.__version__)
print("mps availability: " + str(torch.backends.mps.is_available()))
print("mps build: " + str(torch.backends.mps.is_built()))



model = YOLO("yolov8n.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")# 0代表摄像头
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("traffic.jpeg")
results = model.predict(source=im1, save=True)  # save plotted images
for result in results:
    # Detection
    print(result.boxes.xyxy)   # box with xyxy format, (N, 4)
    print(result.boxes.xywh)   # box with xywh format, (N, 4)
    print(result.boxes.xyxyn)  # box with xyxy format but normalized, (N, 4)
    print(result.boxes.xywhn)  # box with xywh format but normalized, (N, 4)
    print(result.boxes.conf)   # confidence score, (N, 1)
    print(result.boxes.cls)    # cls, (N, 1)

    # Segmentation
    # print(result.masks.data)      # masks, (N, H, W)
    # print(result.masks.xy)        # x,y segments (pixels), List[segment] * N
    # print(result.masks.xyn)       # x,y segments (normalized), List[segment] * N

    # Classification
    # print(result.probs0)     # cls prob, (num_class, )

# from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels，保存预测结果为txt文件

# from list of PIL/ndarray
# results = model.predict(source=[im1, im2])


# This is a sample Python script.