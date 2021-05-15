from copy import deepcopy
import os, ssl
import sys
sys.path.append('../pytorch-image-models-master')
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torch import nn
from torch.nn import *
from torch.nn import functional as F
from torchvision import models
from typing import Optional
from .utils import *
from .triplet_attention import *
from .cbam import *
from .botnet import *
from losses.arcface import ArcMarginProduct
        
import timm
from pprint import pprint

class NFNet(nn.Module):

    def __init__(self, model_name='nfnet_l0', num_class=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, 
        pretrained=True, in_chans=1)
        self.in_features = self.backbone.head.fc.in_features
        self.head = Head(self.in_features,num_class, activation='mish')
        self.out = nn.Linear(self.in_features, num_class)

    def forward(self, x):
        x = self.backbone.stem(x)
        x = self.backbone.stages(x)
        x = self.backbone.final_conv(x)
        x = self.backbone.final_act(x)
        x = self.head(x)
        return x