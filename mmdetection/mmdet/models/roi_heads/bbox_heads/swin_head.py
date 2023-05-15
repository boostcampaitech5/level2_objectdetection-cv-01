import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS


@HEADS.register_module()
class SwinTransformerBBoxHead(nn.Module):
    def __init__(self, num_classes, in_channels, **kwargs):
        super(SwinTransformerBBoxHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.conv_reg = ConvModule(self.in_channels, self.in_channels, 3, stride=1, padding=1)
        self.conv_cls = ConvModule(self.in_channels, self.in_channels, 3, stride=1, padding=1)
        self.conv_output = ConvModule(self.in_channels, self.num_classes * 4, 3, stride=1, padding=1)
    
    def forward(self, inputs):
        cls_feat = inputs
        reg_feat = inputs
        cls_feat = self.conv_cls(cls_feat)
        cls_score = self.conv_output(cls_feat)
        reg_feat = self.conv_reg(reg_feat)
        bbox_pred = self.conv_output(reg_feat)
        return cls_score, bbox_pred