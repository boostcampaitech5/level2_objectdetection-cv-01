import wandb
from mmcv import Config
from mmdet.apis.train import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.utils import get_device


def train(args):
    cfg = Config.fromfile(args.cfg_path + args.model_name + ".py")

    cfg.data.train.classes = args.classes
    cfg.data.train.img_prefix = args.root
    cfg.data.train.ann_file = args.root + args.train_ann_file_name
    cfg.data.train.pipeline[2]["img_scale"] = args.train_resize

    cfg.data.val.classes = args.classes
    cfg.data.val.img_prefix = args.root
    cfg.data.val.ann_file = args.root + args.val_ann_file_name
    cfg.data.val.pipeline[2]["img_scale"] = args.val_resize

    cfg.data.test.classes = args.classes
    cfg.data.test.img_prefix = args.root
    cfg.data.test.ann_file = args.root + args.test_ann_file_name
    cfg.data.test.pipeline[1]["img_scale"] = args.test_resize

    cfg.data.samples_per_gpu = 4

    cfg.seed = 2023
    set_random_seed(cfg.seed, deterministic=False)
    cfg.gpu_ids = [0]
    cfg.work_dir = args.work_dir_path + args.model_name

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    cfg.log_config.hooks = [
        dict(type="TextLoggerHook", interval=100),
        dict(
            type="MMDetWandbHook",
            init_kwargs={"project": args.model_name},
            interval=100,
        ),
    ]
    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)
    model.init_weights()

    train_detector(model, datasets, cfg, distributed=False, validate=True, meta=dict())


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
    train(args=args)
