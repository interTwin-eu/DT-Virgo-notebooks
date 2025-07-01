import torch
from torch.utils.data import DataLoader
import torch.nn as nn
if torch.cuda.is_available():
    device = 'cuda'
    
else:
    device = 'cpu'


import numpy as np
import mlflow
from skimage.measure import label
from tqdm import tqdm
from typing import  Optional, List
import os
from torch.utils.tensorboard import SummaryWriter
import itwinai
from itwinai.components import Predictor, monitor_exec
from itwinai.loggers import  Logger,TensorBoardLogger 
import datetime

from .Model import calculate_iou_2d_non0,generate_data,MeanAbsDiff,StdAbsDiff,calculate_single_loss,ClusterAboveThreshold
from .Data import plot_spectrogram, plot_images,plot_accuracies,plot_cleaned_data,save_tensor

import matplotlib.pyplot as plt


    
    
class GlitchInference (Predictor):
    
    def __init__(self, v_max:int=25,
                 batch_size:int=2,
                 shuffle:bool='False',
                 inference_path:str='./temp/',
                 gen_norm:str|bool=False,n_samp_rows:int=11,logger: Logger | None = None,tensorboard_root:str='/home/jovyan/runs/',
                 trun_name:str='INF/',track_log_freq: int| str = 'batch',ndebug: int=20) -> None:
        
        self.v_max=v_max
        self.logger=logger
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.res_root=inference_path
        self.n_samp_rows=n_samp_rows
        
        self. tensorboard_root=tensorboard_root
        self.trun_name=trun_name
        self.track_log_freq=track_log_freq
        self.ndebug=ndebug
        
    def fraction_empty_lists(self,list_of_lists):
        # Count the number of non-empty lists
        non_empty_count = sum(1 for sublist in list_of_lists if sublist)
    
        # Calculate the fraction
        fraction =1- (non_empty_count / len(list_of_lists))
    
        return fraction
    
    def indices_of_empty_sublists(self,list_of_sublists):
        # Initialize an empty list to store the indices
        indices = []
        non_empty_indices=[]
    
        # Iterate over the elements and their indices
        for i, sublist in enumerate(list_of_sublists):
            if not sublist:  # Check if the sublist is empty
                indices.append(i)  # Append the index to the list
            else:
                non_empty_indices.append(i)
            
    
        return indices,non_empty_indices
    
    def glitch_classifier(self,list_of_lists):
        list=[1 if sublist else 0 for sublist in list_of_lists ]
        #print(len(list))
        return list
    def classifier_accuracy(self,predictions,labels):
        list_check=[(x + y)%2 for x, y in zip(predictions, labels)]
        list_check=np.array(list_check)
        accuracy=1-np.mean(list_check)
        return accuracy
    
    def confusion_matrix(self,predictions,labels):
        cm={}
        for x,y in zip(predictions,labels):
            if x==0:
                if y==0:
                    cm['TN'] = cm.get('TN', 0) + 1
                elif y==1:
                    cm['TP'] = cm.get('TP', 0) + 1
            elif x==1:
                if y==0:
                    cm['FP'] = cm.get('FP', 0) + 1
                elif y==1:
                    cm['FN'] = cm.get('FN', 0) + 1
        return cm


    def roc_curve(self,generated,labels,threshold_set=(10,20.1,0.1),min_cluster_area=10):
    
        roc_dict={}
        for threshold in tqdm(np.arange(threshold_set[0],threshold_set[1],threshold_set[2])):
            try:
                del cluster_nn
            except:
                pass
            cluster_nn= ClusterAboveThreshold(threshold, min_cluster_area).to('cpu')  
            predictions=self.glitch_classifier(cluster_nn(generated))
            cm=self.confusion_matrix(predictions,labels)
            roc_dict[threshold]=cm
        return roc_dict
    
    def analyze_clusters_for_thresholds(self,abs_difference_tensor, generated_tensor, target_tensor, norm_factor, min_cluster_area=1):
        """
        Analyzes cluster data for a range of threshold values.

        Args:
         abs_difference_tensor: Tensor of absolute differences.
         generated_tensor: Tensor of generated data.
         target_tensor: Tensor of target data.
         norm_factor: Normalization factor.
         min_cluster_area: Minimum cluster area.

        Returns:
         A tuple containing two lists:
            - cluster_abs_diff_accuracies: List of classifier accuracies for abs_difference_tensor.
            - clusters_generated_accuracies: List of classifier accuracies for generated_tensor.
       """

        cluster_abs_diff_accuracies = []
        clusters_generated_accuracies = []

        for threshold in tqdm(range(1, 51)):
            try:
                del cluster_nn
            except:
                pass

            # pipeline
            cluster_nn = ClusterAboveThreshold(threshold, min_cluster_area).to('cpu')  # Assuming to('cpu') is needed

            # get clusters
            clusters_abs_diff = cluster_nn(abs_difference_tensor)
            clusters_generated = cluster_nn(generated_tensor * norm_factor)
            clusters_target = cluster_nn(target_tensor * norm_factor)


            #set labels
            target_labels = self.glitch_classifier(clusters_target)  # Use target clusters as labels
            #print("TEST")
            #print("TEST")
            #print("TEST")
            #print(target_labels)
            #print(len(target_labels))
            
            diff_labels= [0 for k in range(len(target_labels))]
        
            # Calculate classifier accuracy for abs_difference_tensor
            abs_diff_predictions = self.glitch_classifier(clusters_abs_diff)
            abs_diff_accuracy = self.classifier_accuracy(abs_diff_predictions, diff_labels)
            cluster_abs_diff_accuracies.append(abs_diff_accuracy)

            # Calculate classifier accuracy for generated_tensor
            generated_predictions = self.glitch_classifier(clusters_generated)
            generated_accuracy = self.classifier_accuracy(generated_predictions, target_labels)
            clusters_generated_accuracies.append(generated_accuracy)

        return cluster_abs_diff_accuracies, clusters_generated_accuracies
        
        
        
    @monitor_exec
    def execute(self, datalist: List) -> int:
        
        channel_means=datalist[0]
        norm_factor=torch.tensor(channel_means[0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        background=datalist[1]
        test=datalist[2]
        gps=datalist[4]
        model_name=datalist[5]
       
        generator_2d=datalist[3].to(device)#model to gpu
        num_aux_channels=test.shape[1]-1
        
        traintms=str(datetime.datetime.now())
        
        tracking_logger=TensorBoardLogger(self.tensorboard_root+self.trun_name+traintms+'-'+model_name,self.track_log_freq)
        
        
        self.logger.create_logger_context()
        tracking_logger.create_logger_context()
        print('Background','\n')
        print(background.shape)
        
        
        test_dataloader = DataLoader(
           test,
           batch_size=self.batch_size,
           shuffle=self.shuffle,
        )
        
        test_background_dataloader = DataLoader(
           background,
           batch_size=self.batch_size,
           shuffle=self.shuffle,
        )
        
        
        print('Generating data...')
        
        for batch in(tqdm(test_dataloader)):
            generated_post=generate_data(generator_2d,batch)
            break
            
        print('generated ',generated_post.shape)    
            
            
        #print('batch ',batch.shape)    
        #torch.save(batch,'./temp/batch.pt')   
        
            
        qplt_g=generated_post[0,0].detach().cpu().numpy()
        qplt_r=batch[0,0].detach().cpu().numpy()
        
        metric_mean=MeanAbsDiff()
        metric_std=StdAbsDiff()
        
        mean_total_loss_train,train_total_loss=calculate_single_loss(generator_2d,metric_mean,test_dataloader)
        
        self.logger.log(
                            item= mean_total_loss_train,
                            identifier='mean_total_loss_train',
                            kind="metric"
                        )
        
        
        
        
        
        
        plot_images(generated_post, batch, channel_means, self.tensorboard_root+self.trun_name,tracking_logger
                    ,gps,num_aux_channels=num_aux_channels,num_images=self.n_samp_rows)
        
        
        torch.cuda.empty_cache()
        
        
        generated_test = torch.tensor([]).to('cpu')  # Initialize an empty tensor
        for batch in tqdm(test_dataloader):
            generated_post = generate_data(generator_2d, batch.detach().cpu()).to('cpu')
            generated_test = torch.cat((generated_test, generated_post), dim=0)
            
            
            
        
        
        
        print(generated_test.shape)
        diff=torch.abs(generated_test-test[:,0,:,:].unsqueeze(1))
        
        npix=torch.sum(diff, dim=(2, 3)) #add mask
        
    
        abs_difference_test=diff*norm_factor
        
        #torch.abs((generated_test-test[:,0,:,:].unsqueeze(1))*norm_factor)
        
        cluster_test = ClusterAboveThreshold(16, 1).to('cpu')
        
        clusters_abs_test = cluster_test(abs_difference_test)
        
        
        
        #print(torch.sum(abs_difference_test, dim=(2, 3)))  
        snr2=abs_difference_test.amax(dim=(1,2,3))
        npix=torch.sum(abs_difference_test, dim=(2, 3)) #add mask
        
        
        cluster_mask= self.glitch_classifier(clusters_abs_test)
        
        cluster_mask=[x==1 for x in cluster_mask]
        
        #print(cluster_mask)
        
        idstep=0
        
        table_md = "| Step | GPS | Cleaned | SNR^2 | N. Pixel |\n|----|----|-------|----------|--------|\n"
        for id_value, bool_value, snr2_val, pix_val in zip(gps,cluster_mask,snr2,npix):
            table_md += f"| {idstep} |{id_value} | {'No' if bool_value else 'Yes'} | {snr2_val: .1f} | {pix_val.int().item() if bool_value else 0 } |\n"
            idstep+=1
        
        tracking_logger.log(table_md,f'Cleaned Batch',kind="text")
        
        selected_gps = [sublist for sublist, mask in zip(gps, cluster_mask) if  mask]
        
        print(len(selected_gps))
        
        cluster_mask=torch.tensor(cluster_mask)
        
        
        
        selected_test=test[cluster_mask]
        
        #print(selected_test.shape)
        
        #print('TEST')
        selected_gen=generated_test[cluster_mask]
        print(selected_gen.shape)
        
        
        plot_images(selected_gen, selected_test, channel_means, self.tensorboard_root+self.trun_name,tracking_logger
                    ,selected_gps,num_aux_channels=num_aux_channels,num_images=self.ndebug,title='Debug')
        
        
        
        
        
       
        print('Assessing accuracy...')  
        
        generated_tensor_pre = torch.tensor([]).to('cpu')  # Initialize an empty tensor
        for batch in tqdm(test_dataloader):
            generated_post = generate_data(generator_2d, batch.detach().cpu()).to('cpu')
            generated_tensor_pre = torch.cat((generated_tensor_pre, generated_post), dim=0)
            
        background_tensor = torch.tensor([]).to('cpu')  # Initialize an empty tensor
        for batch in tqdm(test_background_dataloader):
            background_post = generate_data(generator_2d, batch.detach().cpu()).to('cpu')
            background_tensor = torch.cat((background_tensor, background_post), dim=0)
            
        generated_tensor=torch.cat((generated_tensor_pre,background_tensor), dim=0)    
        #labels=torch.cat((torch.ones(generated_tensor_pre.shape[0]),torch.zeros(background_tensor.shape[0])))    
        target_tensor=torch.cat((test[:,0,:,:].unsqueeze(1),background[:,0,:,:].unsqueeze(1)))
        
        abs_difference_tensor=torch.abs((generated_tensor-target_tensor)*norm_factor)
        
        
        print('ABS TENSOR',abs_difference_tensor.shape)
        
        cluster_abs_diff_accuracies, clusters_generated_accuracies = self.analyze_clusters_for_thresholds(abs_difference_tensor,                                                                                                                         generated_tensor,target_tensor, norm_factor)
        
        
        idstep=1
        
        for cluster_abs_diff_accuracies1, clusters_generated_accuracies1 in zip(cluster_abs_diff_accuracies,clusters_generated_accuracies):
                     tracking_logger.log(cluster_abs_diff_accuracies1,f"Denoising Accuracy VS SNR^2 ",step=idstep)
                     tracking_logger.log(clusters_generated_accuracies1,f"Veto Accuracy VS SNR^2 ",step=idstep)
                     idstep+=1
       
        
        
        #accuracy_gcf=plot_accuracies(cluster_abs_diff_accuracies, clusters_generated_accuracies,self. tensorboard_root+self.trun_name)
        
        #tracking_logger.log(accuracy_gcf,"Accuracy",kind="figure")
        
        
        
        
        empty_idx, non_empty_idx = self.indices_of_empty_sublists(cluster_abs_diff_accuracies)
        
        print( 'ABS',len(cluster_abs_diff_accuracies))
        
        
        
        
        
        iou=calculate_iou_2d_non0(generated_tensor*norm_factor, target_tensor*norm_factor, norm=norm_factor,threshold=15*norm_factor)
        
        
       
        
        self.logger.log(
                            item=iou.item(),
                            identifier='iou_generated',
                            kind="metric"
                        )
        
        
        self.logger.destroy_logger_context()
        
       
        
        
        
        
        return 0
    
    
    
class Glitchflow (Predictor):
    
    def __init__(self,
                 batch_size:int=2,
                 shuffle:bool='False',
                 inference_path:str='./temp/',
                 save_path:str='./temp',
                 fname:str='glitch_generated') -> None:
        
        
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.res_root=inference_path
        self.save_path=save_path
        self.fname=fname
        
        
        
        
    
        
        
        
    @monitor_exec
    def execute(self, datalist: List) -> int:
        
        channel_means=datalist[0]
        norm_factor=torch.tensor(channel_means[0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        background=datalist[1]
        test=datalist[2]
        gps=datalist[4]
        model_name=datalist[5]
       
        generator_2d=datalist[3].to(device)#model to gpu
        num_aux_channels=test.shape[1]-1
        
        traintms=str(datetime.datetime.now())
        
        
        
        test_dataloader = DataLoader(
           test,
           batch_size=self.batch_size,
           shuffle=self.shuffle,
        )
        
        test_background_dataloader = DataLoader(
           background,
           batch_size=self.batch_size,
           shuffle=self.shuffle,
        )
        
        
        
        
        generated_tensor_pre = torch.tensor([]).to('cpu')  # Initialize an empty tensor
        for batch in tqdm(test_dataloader):
            generated_post = generate_data(generator_2d, batch.detach().cpu()).to('cpu')
            generated_tensor_pre = torch.cat((generated_tensor_pre, generated_post), dim=0)
            
        background_tensor = torch.tensor([]).to('cpu')  # Initialize an empty tensor
        for batch in tqdm(test_background_dataloader):
            background_post = generate_data(generator_2d, batch.detach().cpu()).to('cpu')
            background_tensor = torch.cat((background_tensor, background_post), dim=0)
            
        generated_tensor=torch.cat((generated_tensor_pre,background_tensor), dim=0) 
        
        save_tensor( generated_tensor,self.save_path,self.fname)
        
        
       
        
        
        
        
        return 0    



