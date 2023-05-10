_base_ = "../_base_/my_modes/cascade_rcnn_r50_fpn_50e_coco.py"
lr_config = dict(step=[16, 19])
runner = dict(type="EpochBasedRunner", max_epochs=50)
