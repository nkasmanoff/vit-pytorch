"""
PyTorch Lightning wrapper for ResNet18

"""


from pytorch_lightning import Trainer
#from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
#from
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl



import sys
sys.path.append('..')
sys.path.append('../vit_pytorch/')
sys.path.append('utils/')
sys.path.append('data/')
sys.path.append('models/')

from vit import ViT
from recorder import Recorder # import the Recorder and instantiate
from dataloader import *
import cv2

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from argparse import ArgumentParser, Namespace
from test_tube import HyperOptArgumentParser, SlurmCluster
import os
import random
import sys
import resnet
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns




class CNN_Trainer(pl.LightningModule):
    def __init__(self, hparams=None):
        super(CNN_Trainer,self).__init__()


        self.__check_hparams(hparams)
        self.hparams = hparams
        self.prepare_data()

        self.__model = resnet.ResNet18(num_classes=self.num_classes)



    def forward(self,x):

        y_pred = self.__model(x)# returns the predicted class for this dataset.

        return y_pred


    def _run_step(self, batch, batch_idx,step_name):

        img, y_true  = batch
        y_pred = self(img)

        if batch_idx % 1500 == 0:
            # log progress. save a few images from the batch, what they are, and what their prediction is.
            self.__log_step(img,y_true,y_pred, step_name)



        loss = F.cross_entropy(y_pred, y_true)

        return loss , y_pred, y_true


    def training_step(self, batch, batch_idx):

        train_loss, _, _ = self._run_step( batch, batch_idx,step_name='train')
        train_tensorboard_logs = {'train_loss': train_loss}

        return {'loss': train_loss, 'log': train_tensorboard_logs}


    def validation_step(self, batch, batch_idx):

        val_log_dict = {}
        val_loss, y_pred, y_true = self._run_step(batch, batch_idx, step_name='valid')
        y_pred = y_pred.argmax(dim=1).detach().cpu()
        y_true = y_true.detach().cpu()
        val_log_dict['val_loss'] = val_loss
        val_acc = torch.from_numpy(np.array([accuracy_score(y_pred,y_true)]))
        val_log_dict['val_acc'] = val_acc

        return val_log_dict


    def validation_epoch_end(self, outputs):

        val_tensorboard_logs = {}
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        val_tensorboard_logs['avg_val_acc'] = avg_val_acc
        val_tensorboard_logs['avg_val_loss'] = avg_val_loss

        return {'val_loss': avg_val_loss, 'log': val_tensorboard_logs}

    def test_step(self, batch, batch_idx):

        test_log_dict = {}
        test_loss, y_pred, y_true = self._run_step(batch, batch_idx, step_name='test')
        y_pred = y_pred.argmax(dim=1).detach().cpu()
        y_true = y_true.detach().cpu()
        test_log_dict['test_loss'] = test_loss
        test_acc = torch.from_numpy(np.array([accuracy_score(y_pred,y_true)]))
        test_log_dict['test_acc'] = test_acc

        return test_log_dict


    def test_epoch_end(self, outputs):

        test_tensorboard_logs = {}
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        test_tensorboard_logs['avg_test_acc'] = avg_test_acc
        test_tensorboard_logs['avg_test_loss'] = avg_test_loss

        return {'test_loss': avg_test_loss, 'log': test_tensorboard_logs}

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr = self.learning_rate ,weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 4)
        return [optimizer], [scheduler]

    def prepare_data(self):
        # the dataloaders are run batch by batch where this is run fully and once before beginning training
        self.train_loader, self.valid_loader, self.test_loader = get_CIFAR_data(batch_size=self.batch_size,
                                                                                 dset = self.dataset,
                                                                                 )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    def __log_step(self,img, y_true, y_pred, step_name, limit=1):
        ## Plot image
        j = 0 # using the jth element from that batch

        im = img[j].cpu().numpy().transpose(1,2,0)
        # TODO - can say what this was predicted as, compared to ground truth.
        fig, ax = plt.subplots()
        ax.imshow(im) #grayscale
        tag = f'{step_name}_image'
        self.logger.experiment.add_figure(tag, fig, global_step=self.trainer.global_step, close=True, walltime=None)




    def __check_hparams(self, hparams):
        self.channels = hparams.channels if hasattr(hparams, 'channels') else 3
        #self.image_size = hparams.image_size if hasattr(hparams, 'image_size') else 32
        #self.patch_size = hparams.patch_size if hasattr(hparams, 'patch_size') else 8
        #self.depth = hparams.depth if hasattr(hparams, 'depth') else 8
        #self.heads = hparams.heads if hasattr(hparams, 'heads') else 8
        #self.dim = hparams.dim if hasattr(hparams, 'dim') else 768
        #self.mlp_dim = hparams.mlp_dim if hasattr(hparams, 'mlp_dim') else 512
        #self.dropout = hparams.dropout if hasattr(hparams, 'dropout') else 0
        self.num_classes = hparams.num_classes if hasattr(hparams, 'num_classes') else 100

        self.batch_size = hparams.batch_size if hasattr(hparams, 'batch_size') else 128
        self.learning_rate = hparams.learning_rate if hasattr(hparams, 'learning_rate') else 0.001
        self.weight_decay = hparams.weight_decay if hasattr(hparams, 'weight_decay') else 0.001
        self.seed = hparams.seed if hasattr(hparams, 'seed') else 32
        self.dataset = hparams.dataset if hasattr(hparams, 'dataset') else 'cifar100'
        self.architecture = hparams.dataset if hasattr(hparams, 'architecture') else 'ResNet18'



    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False)

        # architecture specific arguments
        parser.add_argument('--channels', type=int, default=3)
        #parser.add_argument('--image_size', type=int, default=32)
        #parser.add_argument('--patch_size', type=int, default=4)  # not really specified
        #parser.add_argument('--depth', type=int, default=12)  # 12, 24, 32
        #parser.add_argument('--heads', type=int, default=12)  # 12, 16, 16
        #parser.add_argument('--dim', type=int, default=768)  # 768, 1024, 1280
        #parser.add_argument('--mlp_dim', type=int, default=3072) # 3072, 4096, 5120
        #parser.add_argument('--dropout', type=float, default=0)  # 0 or .1
        parser.add_argument('--num_classes', type=int, default=10)

        # setup arguments
        parser.add_argument('--batch_size', type=int, default=128)  # 4096
        parser.add_argument('--learning_rate', type=int, default=.001) # .9, .999 (Adam)
        parser.add_argument('--weight_decay', type=int, default=.001) # .1
        parser.add_argument('--seed', type=int, default = 42) # shuffling samples in data loader
        parser.add_argument('--dataset',type=str, default = 'cifar10') # which data set to train with.
        parser.add_argument('--architecture',type=str, default = 'ResNet18') # which data set to train with.

        return parser





if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = CNN_Trainer.add_model_specific_args(parser)
    args = parser.parse_args()
    model = CNN_Trainer(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
    trainer.test()
