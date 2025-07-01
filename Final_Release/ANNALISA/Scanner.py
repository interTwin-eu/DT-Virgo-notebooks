import torch
from .config import device
    
    
from torch.utils.data import DataLoader
import numpy as np


import itwinai
from itwinai.components import Trainer, monitor_exec

from typing import List,Optional 
#another possible types: Tuple Optional


import matplotlib.pyplot as plt

from tqdm import tqdm

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"#VRAM defrag
import time
from datetime import datetime
import yaml


from . Data import TFrame,YamlWrapper,plot_to_f,set_frequency_ticks,plot_stacked_tensor

from . Data import save_tensor, saveYaml

from . Annalisa_gpu import * 



class AnnalisaScan(Trainer):
    def __init__(self,params: YamlWrapper,
                 conf:YamlWrapper,
                 logger: None=None,
                 source:str=''
                 ) -> None:
        
        self.logger=logger
        self.params=params.flist
        self.fconf=conf.flist
        self.source=source
        
        
        
        
        
    def get_or_load(self, dataset: Optional[TFrame] = None):
        """If the dataset is not given, load it from disk."""
        if dataset is None:
            print("WARNING: loading time series dataset from disk.")
            
            tf=TFrame(0,[],[],[],{})
            
            tf.load(self.source)
            
            
            
            return tf
        
        return dataset    
        
        

    @monitor_exec
    def execute(self,pipe_set:Optional[TFrame]=None) -> TFrame:
        
        
        
        
        
        ##############Annalisa parameters ############
        
        proc_set=self.get_or_load(pipe_set)
        data=proc_set.tdata
        tslen=proc_set.tdata.shape[2]
        print(f'Found tensor of shape({proc_set.tdata.shape[0]},{proc_set.tdata.shape[1]},{tslen})')
        sr=proc_set.ann_dict["sample_rate"]
        timestamp=proc_set.ann_dict["id"]
        tag=proc_set.ann_dict["tag"]
        
        
        num_batch=self.params['dataset_scan']['parameters']['num_batch']
        num_chan=self.params['dataset_scan']['parameters']['num_chan']
        refchan=self.params['dataset_scan']['parameters']['refchan']
        threshold=self.params['dataset_scan']['parameters']['threshold']
        
        time_window=self.params['dataset_scan']['parameters']['time_window']
        time_only_mode=self.params['dataset_scan']['parameters']['time_only_mode']
        tolerance_distance=self.params['dataset_scan']['parameters']['tolerance_distance']
        q=self.params['dataset_scan']['parameters']['q']
        frange=self.params['dataset_scan']['parameters']['frange']
        
        fres=self.params['dataset_scan']['parameters']['fres']
        tres=self.params['dataset_scan']['parameters']['tres']
        num_t_bins=self.params['dataset_scan']['parameters']['num_t_bins']
        num_f_bins=self.params['dataset_scan']['parameters']['num_f_bins']
        logf=self.params['dataset_scan']['parameters']['logf']
        qtile=self.params['dataset_scan']['parameters']['qtile']
        whiten=self.params['dataset_scan']['parameters']['whiten']
        
        threshold_corr=self.params['dataset_scan']['parameters']['threshold_corr']
        threshold_iou=self.params['dataset_scan']['parameters']['threshold_iou']
        
        ############################################################################
        
        ################# results######################
        resdir=self.fconf['params']['resdir']
        graphcorr=self.fconf['res']['graphcorr']['file']
        graphiou=self.fconf['res']['graphiou']['file']
        indices=self.fconf['res']['indices']['file']
        annalisaotp=self.fconf['res']['annalisaotp']['file']
        #####################################################
        
        
        ################## dataloader #################
        event_batch_size = num_batch
        channel_batch_size =num_chan
        reference=proc_set.col_list.index(refchan)
        ##########################################
        
        
        
        
        
        
        
        
        
        strain_dataloader = DataLoader(
        data[:,reference,:],
        batch_size=event_batch_size,
        )
        
        
        
       
        aux_dataloader = NestedMultiDimBatchDataLoader(data[:,:,:], event_batch_size, channel_batch_size)
        print('Scanning dataset...')
        
        
        
        annalisa_scan=Annalisa(tslen, sr, device=device, threshold=threshold, time_window=time_window, time_only_mode=time_only_mode,
                 tolerance_distance= tolerance_distance, q=q, frange=frange, fres=fres, tres=tres, num_t_bins=num_t_bins, num_f_bins=num_f_bins,
                 logf=logf, qtile_mode=qtile,whiten=whiten).to(device)
        
        
        
        stacked_corr_coeffs = []
        stacked_iou_coeffs = []

        for batch_s, batch_a in tqdm(zip(strain_dataloader,aux_dataloader),total=len(strain_dataloader)):
            
            iou_coeff_batch,corr_coeff_batch=annalisa_scan(batch_s, batch_a)
            stacked_iou_coeffs.append(iou_coeff_batch)  # Append corr_coeff to the list 
            stacked_corr_coeffs.append(corr_coeff_batch)  # Append corr_coeff to the list  
            
        corr_coeff_dataset=torch.cat(stacked_corr_coeffs, dim=0)
        iou_coeff_dataset=torch.cat(stacked_iou_coeffs, dim=0)
        
        
        #####saving results##############

        column_list=proc_set.col_list
        row_list=proc_set.row_list
        
        timestamp_now=str(int(time.time())*1000)
        
        where=f'{resdir}/{timestamp}_{tag}_{timestamp_now}'
        
        print(where)
        
        os.mkdir(where)
        

        
        corrmean=torch.mean(corr_coeff_dataset,axis=0)
        ioumean=torch.mean(iou_coeff_dataset,axis=0)
        
        
        
        plot_to_f(column_list, 
                            corrmean, 
                            where,
                            graphcorr,
                            title='Correlation histogram vs Hrec')
        
        plot_to_f(column_list, 
                  ioumean,
                  where,
                  graphiou,
                  title='IOU histogram vs Hrec')
        
        
       
        
        
       
        
        mask=(torch.mean(corr_coeff_dataset,axis=0)>threshold_corr) & (torch.mean(iou_coeff_dataset,axis=0)>threshold_iou)
        indices_to_select = torch.nonzero(mask)
        indices_to_select=indices_to_select.flatten()
        
        
        proc_set.addmeta('chans',indices_to_select.tolist())
        print(indices_to_select)

        selected_elements = [column_list[i] for i in indices_to_select]

        print(selected_elements)
        
        save_tensor(indices_to_select,where,indices)
        
        yaml_data = {
         'metadata': {
            'timestamp': datetime.now().isoformat(),
            'id': timestamp,
            'tag': tag 
          },
          'configuration': {
            'parameters':self.params, 
              
            'channels': column_list
          },
          'results': {
            'indeces':indices_to_select.tolist(),  
            'correlations':corrmean.tolist(),
            'iou':ioumean.tolist(),
            'selected': selected_elements    
          }
        }
        
        saveYaml(where,annalisaotp,yaml_data)
                
                
        
        
        
        
        
        
        return proc_set
    
    
    
class Preprocess(Trainer):
    def __init__(self,params: YamlWrapper,conf:YamlWrapper,whiteconf:YamlWrapper,logger: None=None,save:Optional[bool] = False,source:str='',
                 chans:Optional[str]|Optional[List]=None,
                 desired_ticks: List=[8, 20, 30, 50, 100, 200, 500],
                 log_base:float=np.e,
                  phase:bool=False,
                 num_plots:int=10,
                 v_max:int=25,
                 v_min:int=0,length_in_s:int=6,save_gr_par:bool=False) -> None:
        
        self.logger=logger
        self.params=params.flist
        self.fconf=conf.flist
        self.save=save
        self.source=source
        self.chans=chans
        self.whiteconf=whiteconf.flist['whiten']
        self.desired_ticks=desired_ticks
        self.log_base=log_base
        self.phase=phase
        self.num_plots=num_plots
        self.v_max=v_max
        self.v_min=v_min
        self.length_in_s=length_in_s
        self.save_gr_par=save_gr_par
        
    def get_or_load_chs(self, chs: Optional[str]|Optional[List] = None):
        """If the dataset is not given, load it from disk."""
        if type(chs)==str:
            
            channels=torch.load(chs,weights_only=True)
            
            
            
            return channels.tolist()
        
        return chs           
        
    def get_or_load(self, dataset: Optional[TFrame] = None):
        """If the dataset is not given, load it from disk."""
        if dataset is None:
            print("WARNING: loading time series dataset from disk.")
            
            tf=TFrame(0,[],[],[],{})
            
            tf.load(self.source)
            
            return tf
        
        return dataset
    
    
    
        
        
        

    @monitor_exec
    def execute(self,datalist:Optional[TFrame]=None) -> int:
        
        
        ##############Processing parameters space############
        
        proc_set=self.get_or_load(datalist)
        
        if(self.chans):
            indices_to_select=self.get_or_load_chs(self.chans)
            
        else:
            indices_to_select=proc_set.ann_dict["chans"]
            
        print(indices_to_select)    
            
        
        
        data=proc_set.tdata
        rows_list=proc_set.row_list
        column_list=proc_set.col_list
        gps_list=proc_set.gps
        ann_dict=proc_set.ann_dict
        timestamp=proc_set.ann_dict["id"]
        tag=proc_set.ann_dict["tag"]
        tslen=data.shape[2]
        
        print(f'Found tensor of shape({data.shape[0]},{data.shape[1]},{tslen})')
        #indices_to_select=proc_set[1]
        #print(data)
        sr=proc_set.ann_dict["sample_rate"]
        print(sr)
        
        
        num_batch=self.params['dataset_proc']['parameters']['num_batch']
        shuffle=self.params['dataset_proc']['parameters']['shuffle']
        qslicemin=self.params['dataset_proc']['parameters']['qslicemin']
        qslicemax=self.params['dataset_proc']['parameters']['qslicemax']
        
        q=self.params['dataset_proc']['parameters']['q']
        frange=self.params['dataset_proc']['parameters']['frange']
        
        fres=self.params['dataset_proc']['parameters']['fres']
        tres=self.params['dataset_proc']['parameters']['tres']
        num_t_bins=self.params['dataset_proc']['parameters']['num_t_bins']
        num_f_bins=self.params['dataset_proc']['parameters']['num_f_bins']
        logf=self.params['dataset_proc']['parameters']['logf']
        qtile=self.params['dataset_proc']['parameters']['qtile']
        whiten=self.params['dataset_proc']['parameters']['whiten']
        psd=self.params['dataset_proc']['parameters']['psd']
        
        #new params
        energy_mode=self.params['dataset_proc']['parameters']['energy_mode']
        phase_mode=self.params['dataset_proc']['parameters']['phase_mode']
        window_param=self.params['dataset_proc']['parameters']['window_param']
        tau=self.params['dataset_proc']['parameters']['tau']
        beta=self.params['dataset_proc']['parameters']['beta']
        
        
        
        
        ############################################################################
        
        ################# results######################
        resdir=self.fconf['res']['processed']['dir']
        qtensor=self.fconf['res']['qtensor']['file']
        qparameters=self.fconf['res']['qparameters']['file']
        
        #####################################################
        
        
        ################## dataloader #################
        batch_size = num_batch
        
        ##########################################
        
        
        
        
        
        #indices_to_select=torch.flatten(indices_to_select)
        #print(indices_to_select)
        #print(indices_to_select.shape)
        #indices_to_select =torch.tensor([0, 5, 6, 8, 9, 11, 12, 13, 15, 22, 23, 24, 25, 26])
        #print(indices_to_select)
        print('Selected shape:')
        print(data[:,indices_to_select,:].shape)
        
        
    
        
        dataloader = DataLoader(
         data[:,indices_to_select,:],
         batch_size,
         shuffle=shuffle,
        )
        
        
        
        print('Processing...')
        
        qtransform=QT_dataset_custom(tslen, 
                                     sr, 
                                     device=device, 
                                     q=q, 
                                     frange=frange, 
                                     fres=fres, 
                                     tres=tres, 
                                     num_t_bins=num_t_bins, 
                                     num_f_bins=num_f_bins,
                                     logf=logf,
                                     qtile_mode=qtile,
                                     whiten=whiten,
                                     whiteconf=self.whiteconf,
                                     psd=psd,
                                     energy_mode=energy_mode,
                                     phase_mode=phase_mode,
                                     window_param=window_param,
                                     tau=tau,
                                     beta=beta).to(device)
        
  
        
        #qtransform=QT_dataset(6144,1024,device=device,whiten=False,num_t_bins=None,num_f_bins=None).to(device)    
        
        torch.cuda.empty_cache()
        
        qtransform_list=[]
        
        
        #tqdm(zip(strain_dataloader,aux_dataloader),total=len(strain_dataloader)
        for batch in tqdm(dataloader):
             
            with torch.no_grad():
                 transformed= qtransform(batch.to(device=device)).detach().cpu() #torch.Tensor(event.value).to(device)
            print(transformed.shape)   
            qtransform_list.append(transformed[:,:,:,qslicemin:qslicemax])
            del transformed
            torch.cuda.empty_cache()
            gc.collect()
        
        
        stacked_tensor_2d =torch.cat(qtransform_list, dim=0).detach().cpu()
        
        print(stacked_tensor_2d.shape)
        
        
        
        
        #f_range = (8, 410)#take from configuration
        #desired_ticks = [8, 20, 30, 50, 100, 200, 500]
        #log_base = 10  # Or np.e for natural log
        
        timestamp_now=str(int(time.time())*1000)
        
        
        
        where=f'{resdir}/{timestamp}_{tag}_{timestamp_now}'
        
        print(where)
        
        os.mkdir(where)
        
        yaml_data = {
         'metadata': {
            'timestamp': datetime.now().isoformat(),
            'id': timestamp 
          },
          'qt':self.params,
          'whiten':self.whiteconf  
        }
        
        #save_tensor(stacked_tensor_2d,where,qtensor)
        qtf=TFrame(stacked_tensor_2d,rows_list,column_list,gps_list,ann_dict)
        qtf.save(where+"/QT.pt")
        
        print(self.save_gr_par)
        
        
        
        if(self.save_gr_par):
                print('Saving parameters and sample graphs.')
                saveYaml(where,qparameters,yaml_data)
                plot_stacked_tensor(stacked_tensor_2d.unsqueeze(2), 
                           column_list, 
                            frange, 
                            self.desired_ticks, 
                            self.log_base,where,self.phase,
                            self.v_max,self.v_min,self.num_plots,self.length_in_s)
        
         
        
        return 0   
    
    
    
    
