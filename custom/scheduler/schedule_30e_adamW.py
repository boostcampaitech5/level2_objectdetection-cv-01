# optimizer
optimizer = dict(
    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="step", step=[20, 27], warmup_iters=1000)

runner = dict(type="EpochBasedRunner", max_epochs=30)