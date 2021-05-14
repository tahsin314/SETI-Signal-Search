import os
import glob
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
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (ModelCheckpoint, 
LearningRateMonitor, StochasticWeightAveraging,) 
from pytorch_lightning.loggers import WandbLogger
from SETIDataset import SETIDataset, SETIDataModule
from catalyst.data.sampler import BalanceClassSampler
from losses.regression_loss import *
from losses.focal import criterion_margin_focal_binary_cross_entropy, FocalCosineLoss
from utils import *
from model.effnet import EffNet
from model.resne_t import (Resne_t, 
TripletAttentionResne_t, AttentionResne_t, 
CBAttentionResne_t, BotResne_t)
from model.hybrid import Hybrid
from model.vit import ViT
from optimizers.over9000 import AdamW, Ralamb
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
  base = EffNet(pretrained_model=pretrained_model, num_class=num_class, freeze_upto=freeze_upto).to(device)
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

train_ds = SETIDataset(train_df.id.values, train_df.target.values, transforms=train_aug)

if balanced_sampler:
  print('Using Balanced Sampler....')
  train_loader = DataLoader(train_ds,batch_size=batch_size, sampler=
  BalanceClassSampler(labels=train_ds.get_labels(), mode="upsampling"), shuffle=False, num_workers=4)
else:
  train_loader = DataLoader(train_ds,batch_size=batch_size, 
  shuffle=True, drop_last=True, num_workers=4)

valid_ds = SETIDataset(valid_df.id.values, valid_df.target.values, transforms=val_aug)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4)

test_ds = SETIDataset(test_df.id.values, None, transforms=val_aug)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

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
criterion = criterion_margin_focal_binary_cross_entropy

lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
cyclic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer(plist, lr=learning_rate), 
5*len(train_loader), 2, learning_rate/5, -1)


class LightningSETI(pl.LightningModule):
  def __init__(self, model, loss_fn, optim, plist, 
  batch_size, lr_scheduler, random_id, distributed_backend='dp',
  cyclic_scheduler=None, num_class=1, patience=3, factor=0.5,
   learning_rate=1e-3):
      super().__init__()
      self.model = model
      self.num_class = num_class
      self.loss_fn = loss_fn
      self.optim = optim
      self.plist = plist 
      self.lr_scheduler = lr_scheduler
      self.cyclic_scheduler = cyclic_scheduler
      self.random_id = random_id
      self.distributed_backend = distributed_backend
      self.patience = patience
      self.factor = factor
      self.learning_rate = learning_rate
      self.batch_size = batch_size
      self.epoch_end_output = [] # Ugly hack for gathering results from multiple GPUs
  
  def forward(self, x):
      out = self.model(x)
      out = out.type_as(x)
      return out

  def configure_optimizers(self):
        optimizer = self.optim(self.plist, self.learning_rate)
        lr_sc = self.lr_scheduler(optimizer, mode='max', factor=0.5, 
        patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
        return ({
       'optimizer': optimizer,
       'lr_scheduler': lr_sc,
       'monitor': 'val_roc_auc',
       'cyclic_scheduler': self.cyclic_scheduler}
        )
 
  def loss_func(self, logits, labels):
      return self.loss_fn(logits, labels)
  
  def step(self, batch):
    _, x, y = batch
    x, y = x.float(), y.float()
    logits = self.forward(x)
    loss = self.loss_func(torch.squeeze(logits), torch.squeeze(y))
    return loss, logits, y  
  
  def training_step(self, train_batch, batch_idx):
    loss, _, _ = self.step(train_batch)
    self.log('train_loss', loss)
    if self.cyclic_scheduler is not None:
      self.cyclic_scheduler.step()
    return loss

  def validation_step(self, val_batch, batch_idx):
      loss, logits, y = self.step(val_batch)
      self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True) 
      val_log = {'val_loss':loss, 'probs':logits, 'gt':y}
      self.epoch_end_output.append({k:v.cpu() for k,v in val_log.items()})
      return val_log

  def test_step(self, test_batch, batch_idx):
      data_id, x = test_batch
      data_id = [i.split('/')[-1].split('.')[0] for i in list(data_id)]
      pred = self.forward(x)
      pred = torch.softmax(pred, 1)[:,1].detach().cpu().numpy()
      test_log = {'id':data_id, 'target':pred}
      self.epoch_end_output.append({k:v for k,v in test_log.items()})
      return test_log

  def label_processor(self, probs, gt):
    pr = torch.softmax(probs, 1)[:,1].detach().cpu().numpy()
    la = torch.argmax(gt, 1).cpu().numpy()
    return pr, la

  def distributed_output(self, outputs):
    if torch.distributed.is_initialized():
      print('TORCH DP')
      torch.distributed.barrier()
      gather = [None] * torch.distributed.get_world_size()
      torch.distributed.all_gather_object(gather, outputs)
      outputs = [x for xs in gather for x in xs]
    return outputs

  def epoch_end(self, mode, outputs):
    if self.distributed_backend:
      outputs = self.epoch_end_output
    avg_loss = torch.Tensor([out[f'{mode}_loss'].mean() for out in outputs]).mean()
    probs = torch.cat([torch.tensor(out['probs']) for out in outputs], dim=0)
    gt = torch.cat([torch.tensor(out['gt']) for out in outputs], dim=0)
    pr, la = self.label_processor(torch.squeeze(probs), torch.squeeze(gt))
    roc_auc = torch.tensor(roc_auc_score(la, pr))
    print(f'Epoch: {self.current_epoch} Loss : {avg_loss.numpy():.2f}, roc_auc: {roc_auc:.4f}')
    logs = {f'{mode}_loss': avg_loss, f'{mode}_roc_auc': roc_auc}
    self.log(f'{mode}_loss', avg_loss)
    self.log( f'{mode}_roc_auc', roc_auc)
    self.epoch_end_output = []
    return pr, la, {f'avg_{mode}_loss': avg_loss, 'log': logs}

  def validation_epoch_end(self, outputs):
    _, _, log_dict = self.epoch_end('val', outputs)
    self.epoch_end_output = []
    return log_dict

  def test_epoch_end(self, outputs):
    if self.distributed_backend:
      outputs = self.epoch_end_output
    # ids, targets = self.epoch_end('test', outputs)
    ids = np.array([out['id'] for out in outputs]).reshape(-1)
    targets = np.array([out['target'] for out in outputs]).reshape(-1)
    # print(ids, targets)
    zippedList =  list(zip(np.concatenate(ids), np.concatenate(targets)))
    temp_df = pd.DataFrame(zippedList, columns = ['id','target'])
    temp_df.to_csv(f'submission.csv', index=False)

data_module = SETIDataModule(train_ds, valid_ds, None, batch_size=batch_size)
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
swa_callback =StochasticWeightAveraging()

trainer = pl.Trainer(max_epochs=n_epochs, precision=16, auto_lr_find=True,  # Usually the auto is pretty bad. You should instead plot and pick manually.
                  gradient_clip_val=100,
                  num_sanity_val_steps=10,
                  profiler="simple",
                  weights_summary='top',
                  accumulate_grad_batches = accum_step,
                  logger=[wandb_logger], 
                  checkpoint_callback=True,
                  gpus=gpu_ids, num_processes=4*len(gpu_ids),
                  stochastic_weight_avg=False,
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
  lr_finder = trainer.tuner.lr_find(model, train_loader, min_lr=1e-6, max_lr=100, num_training=500)
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

trainer.test(model=model2, test_dataloaders=test_loader)
os.system(f"kaggle competitions submit seti-breakthrough-listen -f submission.csv -m 'Val ROC_AUC: {checkpoint_callback2.best_model_score:.4f}'")