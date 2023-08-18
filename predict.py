from ultralytics import YOLO
from PIL import Image

# Load a model
model = YOLO('./runs/detect/train9/weights/last.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
# results = model(['genshin-impact-mimi-tomo-event-1.jpg'])  # return a list of Results objects
im1 = Image.open("swarm.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs