import os
from matplotlib.pyplot import axis
from config import *
import shutil
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm as T
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (ModelCheckpoint, 
LearningRateMonitor, StochasticWeightAveraging,) 
from pytorch_lightning.loggers import WandbLogger
from SETIDataset import SETIDataset, SETIDataModule
from catalyst.data.sampler import BalanceClassSampler
from losses.regression_loss import *
from losses.focal import (criterion_margin_focal_binary_cross_entropy,
FocalLoss, FocalCosineLoss)
from utils import *
from model.effnet import EffNet
from model.resne_t import (Resne_t, 
TripletAttentionResne_t, AttentionResne_t, 
CBAttentionResne_t, BotResne_t)
from model.nfnet import NFNet 
from model.hybrid import Hybrid
from model.vit import ViT
from optimizers.over9000 import AdamW, Ralamb
from SETIModule import LightningSETI
import wandb

seed_everything(SEED)
os.system("rm -rf *.png")
if mode == 'lr_finder':
  wandb.init(mode="disabled")
  wandb_logger = WandbLogger(project="SETI", config=params, settings=wandb.Settings(start_method='fork'))
else:
  wandb_logger = WandbLogger(project="SETI", config=params, settings=wandb.Settings(start_method='fork'))
  wandb.init(project="SETI", config=params, settings=wandb.Settings(start_method='fork'))
  wandb.run.name= model_name

np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

if 'eff' in model_name:
  base = EffNet(pretrained_model=pretrained_model, num_class=num_class).to(device)
elif 'nfnet' in model_name:
  base = NFNet(model_name=pretrained_model, num_class=num_class).to(device)
elif 'vit' in model_name:
  base = ViT(pretrained_model, num_class=num_class) # Not Working 
else:
  if model_type == 'Normal':
    base = Resne_t(pretrained_model, num_class=num_class).to(device)
  elif model_type == 'Attention':
    base = AttentionResne_t(pretrained_model, num_class=num_class).to(device)
  elif model_type == 'Bottleneck':
    base = BotResne_t(pretrained_model, dim=sz, num_class=num_class).to(device)
  elif model_type == 'TripletAttention':
    base = TripletAttentionResne_t(pretrained_model, num_class=num_class).to(device)
  elif model_type == 'CBAttention':
    base = CBAttentionResne_t(pretrained_model, num_class=num_class).to(device)

wandb.watch(base)

skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)
df = pd.read_csv(f'{data_dir}/train_labels.csv')
df['id'] = df['id'].map(lambda x: f"{image_path}/{x[0]}/{x}.npy")
df = df[['id', 'target']]
df['fold'] = np.nan 
X = df['id']
y = df['target']
train_idx = []
val_idx = []
for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    train_idx = train_index
    val_idx = val_index
    df.loc[val_idx, 'fold'] = i

df['fold'] = df['fold'].astype('int')
train_df = df[(df['fold']!=fold)]
valid_df = df[df['fold']==fold]
test_df = pd.read_csv(f'{data_dir}/sample_submission.csv')
test_df['id'] = test_df['id'].map(lambda x: f"{test_image_path}/{x[0]}/{x}.npy")

train_ds = SETIDataset(train_df.id.values, train_df.target.values, dim=sz,
 transforms=train_aug)

if balanced_sampler:
  sampler = BalanceClassSampler
else: None

valid_ds = SETIDataset(valid_df.id.values, valid_df.target.values, dim=sz,
transforms=val_aug)
test_ds = SETIDataset(test_df.id.values, None, dim=sz, transforms=val_aug)

plist = [ 
        {'params': base.backbone.parameters(),  'lr': learning_rate/10},
        {'params': base.head.parameters(),  'lr': learning_rate}
    ]
if model_type == 'TriplettAttention':
  plist += [{'params': base.at1.parameters(),  'lr': learning_rate}, 
  {'params': base.at2.parameters(),  'lr': learning_rate},
  {'params': base.at3.parameters(),  'lr': learning_rate},
  {'params': base.at4.parameters(),  'lr': learning_rate}]

optimizer = AdamW
criterion = nn.BCEWithLogitsLoss(reduction='sum')

lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
data_module = SETIDataModule(train_ds, valid_ds, test_ds,  sampler= sampler, 
batch_size=batch_size)
cyclic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer(plist, lr=learning_rate), 
5*len(data_module.train_dataloader()), 2, learning_rate/5, -1)

if mode == 'lr_finder': cyclic_scheduler = None
model = LightningSETI(base, criterion, optimizer, plist, batch_size, 
lr_reduce_scheduler,num_class, cyclic_scheduler=cyclic_scheduler, learning_rate = learning_rate)
checkpoint_callback1 = ModelCheckpoint(
    monitor='val_loss',
    dirpath='model_dir',
    filename=f"{model_name}_loss",
    save_top_k=1,
    mode='min',
)

checkpoint_callback2 = ModelCheckpoint(
    monitor='val_roc_auc',
    dirpath='model_dir',
    filename=f"{model_name}_roc_auc",
    save_top_k=1,
    mode='max',
)
lr_monitor = LearningRateMonitor(logging_interval='step')
swa_callback = StochasticWeightAveraging()

trainer = pl.Trainer(max_epochs=n_epochs, precision=16, auto_lr_find=True,  # Usually the auto is pretty bad. You should instead plot and pick manually.
                  gradient_clip_val=100,
                  num_sanity_val_steps=10,
                  profiler="simple",
                  weights_summary='top',
                  accumulate_grad_batches = accum_step,
                  logger=[wandb_logger], 
                  checkpoint_callback=True,
                  gpus=gpu_ids, num_processes=4*len(gpu_ids),
                  stochastic_weight_avg=True,
                  auto_scale_batch_size='power',
                  benchmark=True,
                  distributed_backend=distributed_backend,
                  # plugins='deepspeed', # Not working 
                  # early_stop_callback=False,
                  progress_bar_refresh_rate=1, 
                  callbacks=[checkpoint_callback1, checkpoint_callback2,
                  lr_monitor])

if mode == 'lr_finder':
  trainer.train_dataloader = data_module.train_dataloader
  # Run learning rate finder
  lr_finder = trainer.tuner.lr_find(model, data_module.train_dataloader(), min_lr=1e-6, max_lr=500, num_training=1000)
  # Plot with
  fig = lr_finder.plot(suggest=True, show=True)
  fig.savefig('lr_finder.png')
  fig.show()
# Pick point based on plot, or get suggestion
  new_lr = lr_finder.suggestion()
  print(f"Suggested LR: {new_lr}")
  exit()

wandb.log(params)
# with experiment.record(name=model_name, exp_conf=dict(params), disable_screen=True, token='ae914b4ab3de48eb84b3a4a757c928b9'):
trainer.fit(model, datamodule=data_module)
try:
  print(f"Best Model path: {checkpoint_callback2.best_model_path} Best Score: {checkpoint_callback2.best_model_score:.4f}")
except:
  pass
chk_path = checkpoint_callback2.best_model_path
model2 = LightningSETI.load_from_checkpoint(chk_path, model=base, loss_fn=criterion, optim=optimizer,
 plist=plist, batch_size=batch_size, 
lr_scheduler=lr_reduce_scheduler, cyclic_scheduler=cyclic_scheduler, 
num_class=num_class, learning_rate = learning_rate, random_id=random_id)

trainer.test(model=model2, test_dataloaders=data_module.test_dataloader)
os.system(f"kaggle competitions submit seti-breakthrough-listen -f submission.csv -m '{model_name} Fold {fold} Val ROC_AUC: {checkpoint_callback2.best_model_score:.4f}'")