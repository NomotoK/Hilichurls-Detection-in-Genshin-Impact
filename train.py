from ultralytics import YOLO

# Single GPU & CPU training
# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
#
# # Train the model
# results = model.train(data='data.yaml', epochs=10, imgsz=640)

if __name__ == '__main__':
    # MPS training
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model with 2 GPUs
    results = model.train(data='/Users/hailin/Library/CloudStorage/OneDrive-TheUniversityofLiverpool/SURF23-24/Computer Vision-Based Traffic Accident Detection/datasets/hilichurl-detector.v1i.yolov8/data.yaml', epochs=10, imgsz=640, device='mps')
