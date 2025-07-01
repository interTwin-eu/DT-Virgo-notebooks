import torch
from torch.utils.data import random_split
from .config import device

import numpy as np
import mlflow

import itwinai
from itwinai.components import DataGetter,DataSplitter,DataProcessor,  monitor_exec

from typing import List, Optional, Tuple
import os
import time
import datetime
import pandas as pd
from .Data import (construct_dataframe,
                    process_data_one_by_one,
                    convert_to_torch,
                    TFrame,merge_tframes,
                    save_tensor,YamlWrapper,
                    visualize_dataset)



    
    
    
    
class VisualizeData(DataGetter):
    def __init__(self, data_root:str,data_name:str,tensorboard_root:str) -> None:
        
        self.images_dataset=data_root+data_name
        self.dataname=data_name
        self.tensorboard_root=tensorboard_root
        
        
        
    def LoadTensor(self):
        """If the dataset is not given, load it from disk."""
        
        tf=TFrame(0,[],[],[],{})
            
        tf.load(self.images_dataset)
            
        return tf
        
        

    @monitor_exec
    def execute(self) -> int:
        
        qtf=self.LoadTensor()#list
        
        dataset=qtf.tdata
        
        tms=str(datetime.datetime.now())
        
        visualize_dataset(dataset, self.tensorboard_root+self.dataname+'-'+tms)
       
        return 0 
       



class DtsToTensor(DataGetter):
    def __init__(self, dataconf: YamlWrapper,
                 logger: None=None,savepath:str='./datasets/',
                 tag:str='new') -> None:
        
        self.logger=logger
        self.datalist=dataconf.flist
        self.savepath=savepath
        
        
        if not tag:
            tag:str='new'
            
        self.tag=tag
        
        
        
        

    @monitor_exec
    def execute(self) -> TFrame:
         
        #self.logger.log('Reading data...','MESSAGE',kind='text')
        
        tf_list=[]
        
        
        
        for dataset in self.datalist['datasets']:
            
            
            
            path=dataset['path']
            target=dataset['target']
            ch_list=dataset['ch_list']
            
            channel1=dataset['channels']['min']
            channel2=dataset['channels']['max']
            
            event1=dataset['events']['min']
            event2=dataset['events']['max']
            
            f1=dataset['processing']['minf']
            f2=dataset['processing']['maxf']
            sr=dataset['processing']['sr']
            tslen=dataset['processing']['len']
            batchs=dataset['processing']['batch']
            whiten=dataset['processing']['whiten']
            
            if tslen:
                idx0=5
                idx1=5+tslen
                
                tslen=[idx0,idx1]
            
            print(f"Reading{path}/{target}")
            
            df=construct_dataframe(path=path, 
                           channel_list=ch_list, 
                           target_channel=target, 
                           n1_events=event1, 
                           n2_events=event2, 
                           n1_channels=channel1, 
                           n2_channels=channel2, 
                           print_=True, sr=sr,time_window=tslen,low_freq=f1,high_freq=f2,whiten=whiten)
            
            
            
           
            
            tf_list.append(df)
            
        if len(tf_list)>1:
            merged_df = tf_list[0]  
            for tf in tf_list[1:]:  
                 merged_df = pd.merge(merged_df, tf, on='Event ID')
                
                
        else:
            merged_df=tf_list[0]
            
            
        del tf_list     
            
        
        
        
        ids=list(merged_df['Event ID'])
        merged_df=merged_df.drop('Event ID', axis=1)
        chans=list(merged_df.columns)
        gps=ids
        
        proc_sr=merged_df.iloc[1,1].sample_rate.value
        
        merged_df.to_pickle('temp.pkl')
        merged_df=pd.read_pickle('temp.pkl')
        
        os.remove('temp.pkl')
        
        merged_df=process_data_one_by_one(merged_df,torch.tensor)
        df=convert_to_torch(merged_df)
        
        print(df.shape)
        
        del merged_df
        
        
        
        tfm=TFrame(df,ids,chans,gps,{"sample_rate":int(proc_sr)})    
        
        
        timestamp=str(int(time.time())*1000)
        
        tfm.addmeta('id',timestamp)
        tfm.addmeta('tag',self.tag)
        
        if self.savepath:
            
            
            name=f'{timestamp}_{self.tag}.pt'
            print('Saving ',name)
            tfm.save(self.savepath+name)
            
            
        
       
        
        return tfm
    
    



    
    

    
    
    

    
