from ultralytics import YOLO
from wandb.integration.yolov8 import add_callbacks

if __name__ == '__main__':
    # Load pretrained model
    # model = YOLO("yolov8n.pt")
    model = YOLO("yolov8x.pt")

    # hyperparameter
    img_size = 640
    epoch = 100
    optimizer = 'RMSProp' # ['SGD', 'Adam', 'AdamW', 'RMSProp']
    cos_lr = False # Cosine learning rate scheduler
    label_smoothing = 0.0 # 0.1

    # Wandb
    add_callbacks(model, project="level2) Object Detection")

    model.train(data="coco_recycle.yaml", epochs=epoch, batch=16, imgsz=img_size, 
                save=True, pretrained=True, val=True, save_json=True,
                optimizer= optimizer, cos_lr = cos_lr, label_smoothing= label_smoothing)
    model.val()
