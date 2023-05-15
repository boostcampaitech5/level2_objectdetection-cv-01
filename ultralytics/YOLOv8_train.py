from ultralytics import YOLO
from wandb.integration.yolov8 import add_callbacks

if __name__ == '__main__':
    # Load pretrained model
    model = YOLO("yolov8n.pt")
    # model = YOLO("best.pt")

    # Wandb
    add_callbacks(model, project="level2) Object Detection")

    model.train(data="coco_recycle.yaml", epochs=500, batch=16, imgsz=512, save=True)
    
