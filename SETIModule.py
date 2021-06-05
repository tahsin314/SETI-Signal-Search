from random import choices
from config import *
from losses.mix import *
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

class LightningSETI(pl.LightningModule):
  def __init__(self, model, choice_weights, loss_fns, optim, plist, 
  batch_size, lr_scheduler, random_id, fold=0, distributed_backend='dp',
  cyclic_scheduler=None, num_class=1, patience=3, factor=0.5,
   learning_rate=1e-3):
      super().__init__()
      self.model = model
      self.num_class = num_class
      self.loss_fns = loss_fns
      self.optim = optim
      self.plist = plist 
      self.lr_scheduler = lr_scheduler
      self.cyclic_scheduler = cyclic_scheduler
      self.random_id = random_id
      self.fold = fold
      self.distributed_backend = distributed_backend
      self.patience = patience
      self.factor = factor
      self.learning_rate = learning_rate
      self.batch_size = batch_size
      self.choice_weights = choice_weights
      self.criterion = self.loss_fns[0]
      self.train_loss  = 0
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
       'monitor': f'val_roc_auc_fold_{self.fold}',
       'cyclic_scheduler': self.cyclic_scheduler}
        )
 
  def loss_func(self, logits, labels, choice_weights):
      self.criterion = choices(self.loss_fns, weights=choice_weights)[0]
      return self.criterion(logits, labels)
  
  def step(self, batch, choice_weights):
    _, x, y = batch
    x, y = x.float(), y.float()
    if self.criterion == self.loss_fns[1]:
      x, y1, y2, lam = mixup(x, y)
      y = [y1, y2, lam]
    logits = torch.squeeze(self.forward(x))
    loss = self.loss_func(logits, y, choice_weights)
    return loss, logits, y  
  
  def training_step(self, train_batch, batch_idx):
    if self.current_epoch < 4:
      loss, _, _ = self.step(train_batch, [1.0, 0.0])
    else:
        loss, _, _ = self.step(train_batch, self.choice_weights)
    self.train_loss  += loss.detach()
    self.log(f'train_loss_fold_{self.fold}', self.train_loss/batch_idx, prog_bar=True)
    if self.cyclic_scheduler is not None:
      self.cyclic_scheduler.step()
    return loss

  def validation_step(self, val_batch, batch_idx):
      self.train_loss  = 0
      # self.criterion == self.loss_fns[0]
      loss, logits, y = self.step(val_batch, [1.0, 0])
      self.log(f'val_loss_fold_{self.fold}', loss, on_epoch=True, sync_dist=True) 
      val_log = {'val_loss':loss, 'probs':logits, 'gt':y}
      self.epoch_end_output.append({k:v.cpu() for k,v in val_log.items()})
      return val_log

  def test_step(self, test_batch, batch_idx):
      if len(test_batch) == 2:
        data_id, x = test_batch
      elif len(test_batch) == 3:
        data_id, x, y = test_batch
      data_id = [i.split('/')[-1].split('.')[0] for i in list(data_id)]
      pred = self.forward(x)
      pred = pred.sigmoid().detach().cpu().numpy()
      pred = np.nan_to_num(pred, 0.5)
      if len(test_batch) == 2:
        test_log = {'id':data_id, 'target':np.squeeze(pred)}
      if len(test_batch) == 3:
        test_log = {'id':data_id, 'target':np.squeeze(pred), 'label':np.squeeze(y.detach().cpu().numpy())}
      
      self.epoch_end_output.append({k:v for k,v in test_log.items()})
      return test_log

  def label_processor(self, probs, gt):
    pr = probs.sigmoid().detach().cpu().numpy()
    la = gt.detach().cpu().numpy()
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
    pr = np.nan_to_num(pr, 0.5)
    roc_auc = torch.tensor(roc_auc_score(la, pr))
    print(f'Epoch: {self.current_epoch} Loss : {avg_loss.numpy():.2f}, roc_auc: {roc_auc:.4f}')
    logs = {f'{mode}_loss': avg_loss, f'{mode}_roc_auc': roc_auc}
    self.log(f'{mode}_loss_fold_{self.fold}', avg_loss)
    self.log( f'{mode}_roc_auc_fold_{self.fold}', roc_auc)
    self.epoch_end_output = []
    return pr, la, {f'avg_{mode}_loss': avg_loss, 'log': logs}

  def validation_epoch_end(self, outputs):
    _, _, log_dict = self.epoch_end('val', outputs)
    self.epoch_end_output = []
    return log_dict

  def test_epoch_end(self, outputs):
    if self.distributed_backend:
      outputs = self.epoch_end_output
    ids = np.array([out['id'] for out in outputs]).reshape(-1)
    targets = np.array([out['target'] for out in outputs]).reshape(-1)
    try:
      labels = np.array([out['label'] for out in outputs]).reshape(-1)
    except:
      labels = None
    if labels is not None:
      zippedList =  list(zip(ids, targets, labels))
      temp_df = pd.DataFrame(zippedList, columns = ['id','target', 'label'])
      temp_df.to_csv(f'oof_{self.fold}.csv', index=False)
      self.epoch_end_output = []
    else:
      zippedList =  list(zip(np.concatenate(ids), np.concatenate(targets)))
      temp_df = pd.DataFrame(zippedList, columns = ['id','target'])
      temp_df.to_csv(f'submission_{self.fold}.csv', index=False)