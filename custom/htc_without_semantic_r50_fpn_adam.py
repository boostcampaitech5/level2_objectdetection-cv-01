_base_ = "./model/htc_without_semantic_r50_fpn_model.py"

# fp16 settings
fp16 = dict(loss_scale=512.0)
