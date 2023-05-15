# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.core import BitmapMasks, PolygonMasks, find_inside_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.utils import log_img_scale
from .transforms import Mosaic,MixUp
from ..builder import PIPELINES
try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None



@PIPELINES.register_module()
class MixupOrMosaic:
    def __init__(self,ratio=[0.3,0.3,0.4],img_scale=(1024,1024)):
        if isinstance(ratio, list):
            assert mmcv.is_list_of(ratio, float)
            assert len(ratio)==3
            assert sum(ratio) == 1
        else:
            raise ValueError('ratios must be list of 3 floats')
        self.ratio=ratio
        self.mixup = MixUp(img_scale=img_scale)
        self.mosaic = Mosaic(img_scale=(img_scale[0]//2,img_scale[1]//2))
        self.choice = [self.mixup,self.mosaic,None]
    def __call__(self,results):
        action = np.random.choice(self.choice,p=self.ratio)
        print(action)
        if action == None:
            return results
        else:
            return action(results)  

