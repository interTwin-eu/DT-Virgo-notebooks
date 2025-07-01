from torch.utils.data import DataLoader ,Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import numpy as np
import zipfile
import math
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from os import listdir
import h5py as h5
import os
import multiprocessing
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import logging
import datetime


import gwpy
from gwpy.timeseries import TimeSeries


if torch.cuda.is_available():
    device = 'cuda'
    
else:
    device = 'cpu'
    
    
from typing import  Optional, List,Dict,Literal,Any
from itwinai.torch.trainer import  TorchTrainer
from itwinai.distributed import suppress_workers_print
from itwinai.loggers import EpochTimeTracker, Logger,TensorBoardLogger 
from itwinai.torch.config import TrainingConfiguration
#from itwinai.torch.distributed import DeepSpeedStrategy, RayDDPStrategy, RayDeepSpeedStrategy
from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.torch.profiling.profiler import profile_torch_trainer

from .Model import UNet,CustomLoss,train_decoder,calculate_iou_2d_non0,generate_data
from .Data import plot_images ,TFrame

import mlflow

class VirgoTrainingConfiguration(TrainingConfiguration):
    """Virgo TrainingConfiguration"""

    #: Whether to save best model on validation dataset. Defaults to True.
    save_best: bool = True
    #: Loss function. Defaults to "l1".
    loss: Literal["l1", "l2"] = "l1"
    #: Generator to train. Defaults to "unet".
    generator: Literal["simple", "deep", "resnet", "unet"] = "unet"
    
    
class InterfaceDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        
        return x        


class GlitchTrainer(TorchTrainer):
    def __init__(
        self,
        num_epochs: int = 2,
        coptim_betas: List=[0.9,0.999],
        grad_clip: float=5.0,
        save_name: str='model_checkpoint',
        acc_threshold: int=16,
        config: Dict | TrainingConfiguration | None = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = None,#experimental
        checkpoint_path: str = "checkpoints/epoch_{}.pth",
        temp_path: str= "./temp/",
        logger: Logger | None = None,
        track_log_freq: int| str = 'batch',
        acc_freq:int =1,
        random_seed: int | None = 42,
        name: str | None = None,
        validation_every: int = 0,
        output_channels: int=1,
        tensorboard_root:str='/home/jovyan/runs/',
        model_name:str='Unet_attention_trained',
        trun_name:str='TRAIN/',
        **kwargs,
    ) -> None:
        super().__init__(
            epochs=num_epochs,
            config=config,
            strategy=strategy,
            logger=logger,
            random_seed=random_seed,
            name=name,
            validation_every=validation_every,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))
        # Global training configuration

        if isinstance(config, dict):
            config = VirgoTrainingConfiguration(**config)
        self.config = config
        self.coptim_betas=tuple(coptim_betas)
        self.output_channels=output_channels
        self.num_epochs = num_epochs
        self.acc_freq=acc_freq
        self.checkpoints_location = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        self.temp_path=temp_path
        self.save_checkpoint=self.checkpoints_location+save_name+'.checkpoint_epoch_{}.pth'
        self.save_name=save_name
        self.acc_threshold=acc_threshold
        self.grad_clip=grad_clip
        self. tensorboard_root=tensorboard_root
        self.model_name=model_name
        self.trun_name=trun_name
        self.track_log_freq=track_log_freq
    
    #in a distributed environment, model, optimizer and schedeler must be defined using a distributed strategy
    def create_model_loss_optimizer(self) -> None:
        
        print('Initializing Model...')
        
        
        # Select generator
        print('Initializing Model...')
        print(self.num_aux_channels)
        self.generator_2d = UNet(
             input_channels=self.num_aux_channels,
             output_channels=self.output_channels,
             base_channels=64, # Keep channel specification 64
             use_attention=True 
        ).to(device)
        print(self.generator_2d)
        summary(self.generator_2d, input_size=(self.config['batch_size'], self.num_aux_channels,64,64))
        
        self.loss=CustomLoss()
        
        print('betas')
        print(self.coptim_betas)
        
        self.G_optimizer = torch.optim.AdamW(self.generator_2d.parameters(), 
                            lr=self.config['optim_lr'], 
                            weight_decay=self.config['optim_weight_decay'],  # Critical for generalization
                            betas=self.coptim_betas)
        
        
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                          self.G_optimizer, 
                          mode=self.config.cschd_mode, 
                          patience=self.config.cschd_patience,  # Increased from 5
                          factor=self.config.cschd_factor,
                          min_lr=self.config.cschd_min_lr,
                          verbose=self.config.cschd_verbose
                          )
        
        self.traintms=str(datetime.datetime.now())
        
        self.tracking_logger=TensorBoardLogger(self.tensorboard_root+self.trun_name+self.traintms+'-'+self.model_name,self.track_log_freq)
                          
        
        
                
        
                                             
                                        
        
        
    #### Functions must be modified in  a distributed setting
    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset| None ,
        test_dataset: Dataset| None ,
    ) -> None:
        
        print('Creating dataloaders...')
        batch_size=self.config['batch_size']
        shuffle=self.config['shuffle_train']
        
        
        self.acc_batch_size=batch_size
        
        
        
        
        
        
        self.dataloader = DataLoader(
            train_dataset.data,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        
        
        
        
        
    def train(self):
        # Start the timer for profiling
        #
        print('Train')
        best_checkpoint_filename=self.checkpoints_location.format("best")
        
        debugger = logging.getLogger("mlflow")

        # Set log level to debugging
        debugger.setLevel(logging.DEBUG)
        
        train_loss_plot, val_loss_plot=train_decoder(
            self.num_epochs,
            self.generator_2d,
            self.loss,
            self.G_optimizer,
            self.dataloader,
            self.test,
            self.background,
            self.channel_means,
            self.checkpoints_location,
            snr2_threshold=self.acc_threshold,
            max_grad_clip=self.grad_clip,
            scheduler=self.scheduler,
            logger=self.tracking_logger,
            acc_batch=self.acc_batch_size,
            nacc=self.acc_freq)
        
        
        
        
        
       
        
        #Model signature
        #example_batch = next(iter(self.dataloader))
        #input_example=example_batch[0:2,].numpy()
        
        
        
        self.log(
                            item=self.generator_2d,
                            identifier='trained_model',
                            kind="model"
                        )
        
        
        
        
        
        
        
        
        
        
            
    def execute(
        self,dataset:List
        
    ) -> Any:
        
        
        
        torch.save(dataset[1],self.temp_path+'background.pt')
        
        #print(dataset[2])
        
        
        #moved to dataloading step!
        qtf=TFrame(dataset[2],['events'],['channels'],dataset[6],{"tag":"testdata"})
        qtf.save(self.temp_path+'test.pt')
        
        
        
        
        train=InterfaceDataset(dataset[0])#train
        #validation=InterfaceDataset(dataset[1][:dataset[2].shape[0],:,:])#background
        #validation=InterfaceDataset(dataset[1])
        #test=InterfaceDataset(dataset[2])#test
        
        self.test=dataset[2]
        self.background=dataset[1]
        
        self.norm_factor=dataset[3]
        
        self.num_aux_channels=dataset[4]
        self.channel_means=dataset[5]
        
        
         
        torch.save(self.channel_means,self.temp_path+'means.pt')
        
        
        
        
        
        return super().execute(train)    
