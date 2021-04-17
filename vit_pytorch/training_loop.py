"""
PyTorch Lightning wrapper for different ViT models and datasets. 

"""


from pytorch_lightning import Trainer
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ViT_Trainer(pl.LightningModule):
    def __init__(self, hparams=None):
        super(ViT_Trainer,self).__init__()
        self.__check_hparams(hparams)
        self.hparams = hparams
        self.prepare_data()


    def forward(self,x):

        y_pred = self._model(x)# returns the predicted class for this dataset. 

        return y_pred


    def _run_step(self, batch, batch_idx,step_name):

          = batch
          = self(img)
 
        if batch_idx % 1000 == 0:
            # log progress. save a few images from the batch, what they are, and what their prediction is. 
            self.__log_step( step_name)



        loss =

        return loss, #TODO


    def training_step(self, batch, batch_idx):
        train_loss, _, _, _, _, _ = self._run_step(batch, batch_idx, step_name='train')
        train_tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': train_tensorboard_logs}


