import os
import pandas as pd

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.datasets import build_dataloader, build_dataset
from pycocotools.coco import COCO


def inference(args):
    cfg = Config.fromfile(args.cfg_path + args.model_name + ".py")

    epoch = "latest"

    cfg.data.test.classes = args.classes
    cfg.data.test.img_prefix = args.root
    cfg.data.test.ann_file = args.root + args.test_ann_file_name
    cfg.data.test.pipeline[1]["img_scale"] = args.test_resize
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.seed = 2023
    cfg.gpu_ids = [1]
    cfg.work_dir = args.work_dir_path + args.model_name

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    checkpoint_path = os.path.join(cfg.work_dir, f"{epoch}.pth")

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))  # build detector
    checkpoint = load_checkpoint(
        model, checkpoint_path, map_location="cpu"
    )  # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ""
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += (
                    str(j)
                    + " "
                    + str(o[4])
                    + " "
                    + str(o[0])
                    + " "
                    + str(o[1])
                    + " "
                    + str(o[2])
                    + " "
                    + str(o[3])
                    + " "
                )

        prediction_strings.append(prediction_string)
        file_names.append(image_info["file_name"])

    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f"submission_{epoch}.csv"), index=None)
    print(submission.head())


class Args:
    def __init__(self, filename):
        self.data = {}
        with open(filename, "r") as f:
            for line in f:
                name, value = line.strip().replace(" ", "").split("=")

                try:
                    if name in self.data:
                        raise ValueError(
                            f"Key '{name}' already exists in the dictionary"
                        )
                    if value.isdigit():
                        self.data[name] = (int(value), int(value))
                    else:
                        self.data[name] = value
                except ValueError as e:
                    print(f"Error: {e}")
                    exit(1)

        self.data["classes"] = (
            "General trash",
            "Paper",
            "Paper pack",
            "Metal",
            "Glass",
            "Plastic",
            "Styrofoam",
            "Plastic bag",
            "Battery",
            "Clothing",
        )

    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise AttributeError(f"'Data' object has no attribute '{key}'")

    def __str__(self):
        return str(self.data)


if __name__ == "__main__":
    args = Args("./custom/data_config.txt")

    print(args)
    inference(args=args)
