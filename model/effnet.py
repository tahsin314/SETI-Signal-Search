## Code copied from Lukemelas github repository have a look
## https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch
"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""
import warnings
import sys
sys.path.append('../pytorch-image-models-master')
from torch.optim import optimizer
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
import math
import collections
from functools import partial
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
import pytorch_lightning as pl
from .utils import *
from losses.arcface import ArcMarginProduct
from losses.triplet_loss import *
class EffNet(nn.Module):
    def __init__(self, pretrained_model='tf_efficientnet_b4', num_class=1):
        super(EffNet, self).__init__()
        self.backbone = timm.create_model(pretrained_model, pretrained=True, in_chans=1)
        self.in_features = self.backbone.bn2.num_features
        self.head = Head(self.in_features, num_class, activation='mish')
        
    def forward(self, x):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.blocks(x)
        x = self.backbone.conv_head(x)
        output = self.head(x)
        return output
