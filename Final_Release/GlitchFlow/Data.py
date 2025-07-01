import torch
from .config import device
    
    
import pandas as pd

import numpy as np
import os
import json
import yaml
import h5py as h5
from os import listdir
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from scipy.signal import sosfilt_zi
import torchaudio.functional as fn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import datetime

class TFrame:
    
    __slots__=['tdata','row_list','col_list','gps','ann_dict']
    
    def __init__(self,data,rows,cols,gps,anns):
        
        self.tdata=data 
        self.row_list=rows
        self.col_list=cols
        self.gps=gps
        
        #possible annotations: id,sr, u.m.
        self.ann_dict=anns
        #"sample_rate"
        
    def thead(self):
        
        df = pd.DataFrame(self.tdata[:5,:,0].numpy(), index=self.row_list[:5], columns=self.col_list)
        
        return df
    
    def save(self,path):
        
        frame={'data':self.tdata,'rows':self.row_list,'cols':self.col_list,'gps':self.gps,'ann':self.ann_dict}
        
        torch.save(frame,path)
        
    def load(self,path,safeload=True):
        
        output=torch.load(path,weights_only=safeload)
        
        self.tdata=output['data']
        
        
        
        self.row_list=output['rows']
        
        self.col_list=output['cols']
        
        self.gps=output['gps']
        
        self.ann_dict=output['ann']
        
        
        
    
    def addmeta(self,key,annotation):
        
        self.ann_dict.update({key: annotation})
        
    
    


    



def crop(data,clen,srate):
    
    tlen=data.shape[2]*(1/srate)
    
    
    
    
    idx0=int(((tlen-clen)/2)*srate)
    idx1=int(((tlen+clen)/2)*srate)
    
    
    
     
    data=data[:,:,idx0:idx1]
    
    
    
    return data
        




def merge_tframes(tf_list):
    
    comm_ids=set(tf_list[0].row_list)
    for lst in tf_list[1:]:
        comm_ids &=set(lst.row_list)
    
    new_rows=list(comm_ids)
    
    new_cols=[]
    new_cols+=tf_list[0].col_list
    for lst in tf_list[1:]:
        new_cols+=lst.col_list
        
        
        
    temp_list=[]
    stack_list=[]
    for id in new_rows:  
        temp_list=[]
        buff=tf_list[0].tdata[tf_list[0].row_list.index(id),:,:]
        temp_list.append(buff)
        
        for lst in tf_list[1:]:
            buff=lst.tdata[lst.row_list.index(id),:,:]   
            temp_list.append(buff)
            
        row=torch.cat(temp_list,axis=0)
        stack_list.append(row)
   
    dfm=torch.stack(stack_list,axis=0)
    
    tf_stack=TFrame(dfm,new_rows,new_cols,new_rows,{"sample_rate":tf_list[0].ann_dict['sample_rate']})
    
    return tf_stack


def construct_dataframe(path,channel_list=None ,target_channel='V1:Hrec_hoft_16384Hz',n1_events=None, n2_events=None,n1_channels=None,
                        n2_channels=None,print_=True,sr=False,low_freq=4,high_freq=50, time_window=None, whiten=False):
    """
    Construct a DataFrame from data stored in HDF5 files.

    Parameters:
    - path (str): The directory path where the HDF5 files are located.
    - channel_list (list): A list of channel names to include in the DataFrame. If not provided, it defaults to None, and the code will load all channels in the file.
    - target_channel (str): The target channel to include in the DataFrame. Default value is 'V1:Hrec_hoft_16384Hz'.
    - n1_events (int): The starting index of events to consider. Default value is None, which corresponds to first file in directory.
    - n2_events (int): The ending index of events to consider. Default value is None, which corresponds to last file in directory.
    - n1_channels (int): The starting index of channels to consider. Default value is None, which corresponds to first channel in file.
    - n2_channels (int): The ending index of channels to consider. Default value is None, which corresponds to last channel in directory.
    - print_ (bool): A boolean indicating whether to print progress information. Default value is True.
    - sr (float or bool): New sample rate for resampling the data. Default value is False, which stands for no resampling.
    - low_freq (int): lowest frequency allowed by the filter bandpass.
    - high_freq (int): highest frequency allowed by the filter bandpass.
    - time_window (list): list with two elements with the start and end of the cropped timeseries.
    - whiten (bool): value to set if one wants to whiten the timeseries with the GWPy method.

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the HDF5 files.
    """
    
    if not n1_events:
        n1_events=0
    if not n2_events:
        n2_events=len(listdir(path))
        
    if n2_events>len(listdir(path)):
        n2_events=len(listdir(path))
    
    #print(f'PATH: {path}')    
    lstdr=listdir(path)[n1_events:n2_events]
    #print(f'LIST DIR: {lstdr}')
    sample_file=listdir(path)[0]
    print(sample_file)
    
    files = [f for f in lstdr]
    # print(files)
    df_list = []
    event_data = []
    
    
    if not channel_list:
        n_all_channels=0
        all_channels=[]
        with h5.File(os.path.join(path, sample_file), 'r') as fout:
            event_id = list(fout.keys())[0]
            all_channels=list(fout[event_id])
            n_all_channels=len(list(fout[event_id]))
            #print(f'event id: {event_id}')


        if not n1_channels:
            n1_channels=0
        if not n2_channels:
            n2_channels=n_all_channels

        if n2_channels>n_all_channels:
            n2_channels=n_all_channels

        channels=all_channels[n1_channels:n2_channels]
    else:
        channels=channel_list
    try:
        channels.remove(target_channel)
    except:
        pass
    
    sr0=0
    for i, file in enumerate(files):
        if print_:
            print(f"Added {i + 1}/{n2_events - n1_events} files to dataframe", end='\r')
       
           
        try:
            
            with h5.File(os.path.join(path, file), 'r') as fout:
                #print('file successfully opened')
                event_id = list(fout.keys())[0]
                gps_time=fout[event_id][target_channel].attrs['t0']
                dictionary = {'Event ID': event_id}
                #dictionary={}
                event_data.append(event_id)
                
                data=fout[event_id][target_channel]
                sample_rate=data.attrs['sample_rate']
                
                tmsrs = TimeSeries(data,dt=1.0/sample_rate) #ToDo: add other attrs to the TimeSeries (seee h5_gwpy.ipynb for reference)              
                
                bp = filter_design.bandpass(low_freq, high_freq, tmsrs.sample_rate)


                if time_window:
                    tmsrs=tmsrs[int(time_window[0]*sample_rate):int(time_window[-1]*sample_rate)]

                if whiten:
#                    tmsrs=tmsrs.whiten()
                    tmsrs=whiten_(tmsrs)
                

                

                # we need to filter AFTER the whitening
                tmsrs = tmsrs.filter(bp, filtfilt=True)
                

                if sr:
                    tmsrs=tmsrs.resample(sr)
#                    tmsrs=TimeSeries(np.array(tmsrs)/max(np.array(tmsrs)), dt=1/sr)
#                    if time_window:
#                        tmsrs=tmsrs[int(time_window[0]*sr):int(time_window[-1]*sr)]
                
                dictionary[target_channel] = [tmsrs]
                #print(f'DICT: {dictionary}')
        
                for i,channel in enumerate(channels):
                    #print(f"Added {i + 1}/{n2_channels - n1_channels} files to dataframe", end='\r')
                    try:
                        data=fout[event_id][channel]
                        sample_rate=data.attrs['sample_rate']
                        
                        tmsrs = TimeSeries(data,dt=1.0/sample_rate)

                           
                        if time_window:
                            tmsrs=tmsrs[int(time_window[0]*sample_rate):int(time_window[-1]*sample_rate)]
                        

                        if whiten:
#                            tmsrs=tmsrs.whiten()
                            tmsrs=whiten_(tmsrs) 

                        tmsrs = tmsrs.filter(bp, filtfilt=True)
                 
                        if sr:
                            
                            tmsrs=tmsrs.resample(sr)
#                            tmsrs=TimeSeries(np.array(tmsrs)/max(np.array(tmsrs)), dt=1/sr)

#                            if time_window:
#                                tmsrs=tmsrs[int(time_window[0]*sr):int(time_window[-1]*sr)]
                           
                        dictionary[channel] = [tmsrs]
                        
                    except Exception as e:
                        tmsrs=np.nan
                        dictionary[channel] = [tmsrs]
                        #print(f'error in making timeseries:')
                        #print(e)
                        
                
                df_list.append(pd.DataFrame(dictionary))
                    

                
        except Exception as e:
        
            if print_:
                print(f'COULD NOT OPEN {os.path.join(path, file)}')
                print(e)
            
        
        
    #print(f'DF LIST: {df_list.shape}')
    
    df = pd.concat(df_list, ignore_index=True)
    
    #df_ids = pd.DataFrame({'Event ID': event_data})
    #df = pd.concat([df_ids, df], axis=1)
    
    return df 

def process_data_one_by_one(df, func, *args, **kwargs):
    '''
    Apply a processing function to each element of a DataFrame individually, with optional arguments.

    Parameters:
    df (DataFrame): 
        A pandas DataFrame containing numerical data to be processed.
    func (function): 
        A processing function to be applied to each element of the DataFrame. This function should accept at least one argument 
        and optionally additional arguments.
    *args: 
        Optional positional arguments to be passed to func.
    **kwargs: 
        Optional keyword arguments to be passed to func.

    Returns:
    DataFrame
        A new DataFrame where each element has been processed by the given function with the provided arguments.
    '''
    return df.applymap(lambda t: func(t, *args, **kwargs))
    #return df.progress_applymap(lambda t: func(t, *args, **kwargs))   


def convert_to_torch(df):
    '''
    Converts input DataFrame into a torch tensor
    
    Parameters:
    - df (DataFrame): input dataframe to be converted to pytorch tensor
    Returns:
    - torch tensor: tensor containg input data
    '''
    return torch.stack([torch.stack([*df.iloc[i]]) for i in range(df.shape[0])])








def save_tensor(tensor,savepath,name):
    
    torch.save(tensor, f'{savepath}/{name}.pt')
    
    
    
def save_to_json(data, filename, folder="."):
    """
    Save a dictionary or list to a JSON file in the specified folder.
    
    Args:
        data: Dictionary or list to be saved.
        filename (str): Name of the JSON file (without .json extension).
        folder (str): Target folder path (default: current directory).
    
    Returns:
        str: Full path to the saved file.
    
    Raises:
        TypeError: If data is not a dictionary or list.
        OSError: If the folder cannot be created.
    """
    # Validate input type
    if not isinstance(data, (dict, list)):
        raise TypeError("Data must be a dictionary or list.")
    
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Add .json extension if not present
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Build full path
    full_path = os.path.join(folder, filename)
    
    # Write JSON data with pretty formatting
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return full_path





#---------Wrapper-----------------------------------------------------------------------------------

class YamlWrapper:
    def __init__(self, fname:str,path:str='./conf/') -> None:
        self.fpath=path+fname
        self.flist=None
        self.err=None
        
        try:
            with open(self.fpath, 'r') as file:
                self.flist = yaml.safe_load(file)
        except Exception as e:
            self.err=e
            
            
def saveYaml(path,name,struct):
    
    
   
    
    filepath = os.path.join(path, name)
    
    err=None
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
             yaml.dump(
                 struct,
                 f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=80,
                indent=4
             )
                
    except  Exception as e:
             err=e
        
    return err    
            
            
         
    
    


#--------------------------------------Training functions------------------------------------------------------------------------------------------#

def augment_data(tensor, num_slices):
    B, C, H, W = tensor.shape
    W0 = H  # Target width is now H
    offset = (W - num_slices * W0) // 2

    selected_chunks = tensor[:, :, :, offset:offset + num_slices * W0].view(B, C, H, num_slices, W0)
    tensor_permuted = selected_chunks.permute(0, 3, 1, 2, 4)
    augmented_tensor = tensor_permuted.contiguous().view(B * num_slices, C, H, W0)
    return augmented_tensor


def augment_dataset(train_data,test_data):
    
    # Augment training data (3 slices)
    train_data_augmented_3 = augment_data(train_data, 3)

    # Augment training data (2 slices)
    train_data_augmented_2 = augment_data(train_data, 2)

    train_data_2d = torch.cat([train_data_augmented_3, train_data_augmented_2], dim=0)

    # Augment validation data (3 slices)
    val_data_augmented_3 = augment_data(test_data, 3)

    # Augment validation data (2 slices)
    val_data_augmented_2 = augment_data(test_data, 2)

    test_data_2d = torch.cat([val_data_augmented_3, val_data_augmented_2], dim=0)
    
    return train_data_2d, test_data_2d

def augment_list(list0,n_ext=5):
    
    ltemp=[]
    
    for elm in list0:
        temp=[elm]*n_ext
        ltemp.extend(temp)
    

    return ltemp
    
    

"""
def filter_rows_below_threshold(data, threshold):
    
    Filters rows in the data tensor where all channels are below a certain threshold.

    Input:
    - data (torch.Tensor): dataset
    - threshold (torch.Tensor): threshold value for each channel

    Return:
    - filtered_data (torch.Tensor): filtered dataset
    
    # Calculate the maximum value for each channel across all examples
    max_vals = data.view(data.shape[0], data.shape[1], -1).max(-1)[0]
    #print(max_vals.shape)
    #print(threshold.unsqueeze(0).shape)
    # Check if all three values in each row are below the respective threshold
    mask = (max_vals < threshold.unsqueeze(0)).all(dim=1)
    #print(mask.shape)
    
    # Use the boolean mask to filter and keep only the rows in the dataset that satisfy the condition
    filtered_data = data[mask]

    return filtered_data,mask
"""

def filter_rows_below_threshold(data, threshold):
    """
    Filters rows in the data tensor where all channels are below a certain threshold.
    Input:
    - data (torch.Tensor): dataset
    - threshold (torch.Tensor): threshold value for each channel
    Return:
    - filtered_data (torch.Tensor): filtered dataset
    """
    # Calculate the maximum value for each channel across all examples
    max_vals = data.view(data.shape[0], data.shape[1], -1).max(-1)[0]
    max_vals=max_vals.detach().cpu()
    # Check if all three values in each row are below the respective threshold
    mask = (max_vals < threshold.unsqueeze(0).detach().cpu()).all(dim=1)
    # Use the boolean mask to filter and keep only the rows in the dataset that satisfy the condition
    filtered_data = data[mask]
    return filtered_data,mask


def filter_rows_above_threshold(data, threshold):
    """
    Filters rows in the data tensor where all channels are below a certain threshold.

    Input:
    - data (torch.Tensor): dataset
    - threshold (torch.Tensor): threshold value for each channel

    Return:
    - filtered_data (torch.Tensor): filtered dataset
    """
    # Calculate the maximum value for each channel across all examples
    max_vals = data.view(data.shape[0], data.shape[1], -1).max(-1)[0]
    #max_vals=max_vals.detach().cpu()
    #print(max_vals.shape)
    #print(threshold.unsqueeze(0).shape)
    # Check if all three values in each row are below the respective threshold
    mask = (max_vals >= threshold.unsqueeze(0).detach().cpu()).all(dim=1)
    #print(mask.shape)
    
    # Use the boolean mask to filter and keep only the rows in the dataset that satisfy the condition
    filtered_data = data[mask]

    return filtered_data,mask.tolist()

def gen_mask(strain_thr,n_chan,aux_thr):
    
    mask=[]
    
    mask.append(strain_thr)
    
    for i in range(n_chan-1):
        
         mask.append(aux_thr)
            
    print(mask)        
            
            
    return torch.tensor(mask)        
        
    
    


def find_max(data):
    #print(data.shape)
    """
    Normalizes the qplot data to the range [0,1] for NN convergence purposes
    
    Input:
    - data (torch.Tensor) : dataset of qtransforms
    
    Return:
    - data (torch.tensor) : normalized dataset
    """
    max_vals = data.view(data.shape[0], data.shape[1], -1).max(-1)[0]  # Compute the maximum value for each 128x128 tensor
    max_global = data.view(data.shape[0], data.shape[1], -1).max(0)[0].max(1)[0]
    #print(max_global)
    print("Maximum value for each element tensor:", max_vals.shape)
    max_vals = max_vals.unsqueeze(-1).unsqueeze(-1)  # Add dimensions to match the shape of data for broadcasting
    return max_vals


def normalize_ch_mean(data, channel_means, channel_std=None):
    
    #MODIFY WITH MEDIAN
    """
    Normalizes the data by dividing each channel by its respective mean value,
    or by subtracting the mean and dividing by the standard deviation if channel_std is provided.

    Input:
    - data (torch.Tensor): dataset
    - channel_means (list or torch.Tensor): list of mean values for each channel
    - channel_std (list or torch.Tensor, optional): list of standard deviation values for each channel. Defaults to None.

    Return:
    - normalized_data (torch.Tensor): normalized dataset
    """
    # Convert channel_means and channel_std to tensors if they're not already
    if not isinstance(channel_means, torch.Tensor):
        channel_means = torch.tensor(channel_means)
    if channel_std is not None and not isinstance(channel_std, torch.Tensor):
        channel_std = torch.tensor(channel_std)


    # Check if channel_means has the correct shape
    if channel_means.shape[0] != data.shape[1]:
        raise ValueError("Number of elements in channel_means must match the number of channels in data.")

    # Reshape channel_means and channel_std to match the shape of data for broadcasting
    channel_means = channel_means.view(1, -1, 1, 1)
    if channel_std is not None:
        if channel_std.shape[0] != data.shape[1]:
            raise ValueError("Number of elements in channel_std must match the number of channels in data.")
        channel_std = channel_std.view(1, -1, 1, 1)

    # Normalize data
    if channel_std is None:
        normalized_data = data / channel_means
    else:
        normalized_data = (data - channel_means) / channel_std

    return normalized_data

#----------------Visualization------------------------------------------------------------------------------------------------------------------------

def plot_spectrogram(batch,vmin=0,vmax=1):
    
    fig,ax=plt.subplots(figsize= (6, 6))
    
    
    ax.imshow(batch, aspect='auto',vmin=vmin,vmax=vmax)
    ax.set_title('Real')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.colorbar()
    
    
    return fig




def plot_images(generated_post, batch, 
                channel_means, run_name,
                logger,gps,num_aux_channels=8,
                num_images=10,
                v_max=25,
                base_path='./temp',
                title='inference'):
    
    
    #writer = SummaryWriter(run_name)
    
    
    for i in range(num_images):
        print('---------------------------')
        print(f'IMAGE {i}')

        qplt_g = torch.flipud(generated_post[i, 0].detach().cpu() * channel_means[0])
        qplt_r = torch.flipud(batch[i, 0].detach().cpu() * channel_means[0])

        time_extent = generated_post[i, 0].shape[0]
        freq_extent = generated_post[i, 0].shape[1]
        extent = [0, time_extent, 0, freq_extent]

        num_rows_aux = (num_aux_channels + 3) // 4

        fig, axes = plt.subplots(1 + num_rows_aux, 4, figsize=(20, 5 * (1 + num_rows_aux)))

        # Handle the case where there's only one row (including 0 aux channels)
        if 1 + num_rows_aux == 1:  # Only one row
            axes = np.array([axes]) # make axes 2D so that it works with the rest of the code
            axes = axes.reshape(1,4) # reshape it to be a 1x4 array
            
            
        fig.suptitle(f'Event {gps[i]}')
        im_r = axes[0, 0].imshow(qplt_r, aspect='auto', extent=extent, vmin=0, vmax=v_max)
        axes[0, 0].set_title(f'Real')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency')
        fig.colorbar(im_r, ax=axes[0, 0])

        im_g = axes[0, 1].imshow(qplt_g, aspect='auto', extent=extent, vmin=0, vmax=v_max)
        axes[0, 1].set_title('Generated')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Frequency')
        fig.colorbar(im_g, ax=axes[0, 1])

        im_diff = axes[0, 2].imshow(torch.abs(qplt_g - qplt_r), aspect='auto', extent=extent, vmin=0, vmax=v_max)
        axes[0, 2].set_title('True - Generated')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Frequency')
        fig.colorbar(im_diff, ax=axes[0, 2])

        axes[0, 3].axis('off')

        aux_channel_index = 1
        row = 1
        col = 0

        for j in range(num_aux_channels):
            qplt_aux = torch.flipud(batch[i, aux_channel_index].detach().cpu() * channel_means[aux_channel_index])
            im_aux = axes[row, col].imshow(qplt_aux, aspect='auto', extent=extent, vmin=0, vmax=v_max)
            axes[row, col].set_title(f'aux{aux_channel_index}')
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel('Frequency')
            fig.colorbar(im_aux, ax=axes[row, col])

            aux_channel_index += 1
            col += 1
            if col == 4:
                col = 0
                row += 1

        for r in range(row, 1 + num_rows_aux):
            for c in range(4):
                axes[r, c].axis('off')

        plt.tight_layout()
        logger.log(plt.gcf(),title,kind="figure",step=i)
        #plt.savefig(f'{base_path}/inference{i}.pdf')
        #writer.add_figure(f'inference{i}_{datetime.datetime.now()}', plt.gcf())
        #writer.flush()
        
        
def plot_accuracies(cluster_abs_diff_accuracies, clusters_generated_accuracies,run_name,th0=1,th1=51, base_path='./temp'):
    
        #writer = SummaryWriter(run_name)
        
        thresholds = range(th0, th1)  # The SNR^2 thresholds

        plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization

        plt.plot(thresholds, cluster_abs_diff_accuracies, label="Denoising Accuracy", marker='o', linestyle='-')
        plt.plot(thresholds[5:], clusters_generated_accuracies[5:], label="Vetoing Accuracy", marker='x', linestyle='--')
        #plt.plot(thresholds, cluster_abs_diff_accuracies_veto, label="Vetoing Accuracy for veto correctly flagged data", marker='p', linestyle='--')

        plt.xlabel(r"$\mathrm{SNR^2}$ Threshold", fontsize=20)
        plt.ylabel("Accuracy", fontsize=20)
        plt.title("Accuracy vs. $\mathrm{SNR^2}$", fontsize=22)
        plt.xticks(np.arange(min(thresholds), max(thresholds)+1, 5.0), fontsize=16) # set ticks every 5
        plt.yticks(fontsize=16)
        plt.grid(True)  # Add a grid for better readability
        plt.legend(fontsize=20)  # Show the legend
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.savefig(f'{base_path}/accuracy.png')
        
        return plt.gcf()
        
        #writer.add_figure(f'accuracy_{datetime.datetime.now()}', plt.gcf())
        #writer.flush()
        
        
        
def set_frequency_ticks(ax, f_range, desired_ticks, log_base, new_height):
    """Sets the y-axis (frequency) ticks and labels."""
    log_f_range = (np.log(f_range[0]) / np.log(log_base), np.log(f_range[1]) / np.log(log_base))
    log_desired_ticks = np.log(desired_ticks) / np.log(log_base)

    y_ticks_pixel = np.interp(log_desired_ticks, log_f_range, [new_height - 1, 0])

    y_ticks_pixel = [int(p) for p in y_ticks_pixel]
    y_ticks_pixel = np.clip(y_ticks_pixel, 0, new_height - 1)

    y_ticks_pixel, unique_indices = np.unique(y_ticks_pixel, return_index=True)
    desired_ticks_used = np.array(desired_ticks)[unique_indices].tolist()

    ax.grid(True, axis='y', which='both')
    ax.set_yticks(y_ticks_pixel)
    ax.set_yticklabels(np.flipud(desired_ticks_used),fontsize=16)
    ax.invert_yaxis() # Important: Invert y-axis for spectrograms
    
    
def plot_cleaned_data(empty_idx,non_empty_idx,target_tensor,generated_tensor,abs_difference_tensor,norm_factor,run_name,
                      base_path='./temp',
                      v_max = 25,
                      f_range = (8, 500),
                      desired_ticks = [8, 20, 30, 50, 100, 200, 500],
                      log_base = 10):
    
    j=0
    
    writer = SummaryWriter(run_name)
    
    for i in non_empty_idx:
    
        if j == 30:
            break
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Define fig and axes HERE   
        
        # Plotting
        im0 = axes[0].imshow((target_tensor * norm_factor)[i].squeeze(0), cmap='viridis', vmin=0, vmax=v_max, aspect='auto')
        axes[0].set_title('Target',fontsize=22)
        im1 = axes[1].imshow((generated_tensor * norm_factor)[i].squeeze(0), cmap='viridis', vmin=0, vmax=v_max, aspect='auto')
        axes[1].set_title('Generated',fontsize=22)
        im2 = axes[2].imshow((abs_difference_tensor)[i].squeeze(0), cmap='viridis', vmin=0, vmax=v_max, aspect='auto')  # Store the image for cbar
        axes[2].set_title('Cleaned',fontsize=22)
        
        for ax in axes: # Apply frequency ticks to all subplots
            set_frequency_ticks(ax, f_range, desired_ticks, log_base, target_tensor.shape[-2]) # Use target_tensor or generated_tensor shape
            ax.set_xticks([0, 31, 63])
            ax.set_xticklabels([0, 0.5, 1],fontsize=16)
            ax.set_xlabel("Time (s)",fontsize=20) # Add X label
            ax.set_ylabel("Frequency (Hz)",fontsize=20) # Add X label
            
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)  # Adjust fraction and pad as needed
            
        plt.tight_layout() # Adjust subplot params for a tight layout
        plt.savefig(f'{base_path}/cleaned_{i}.png')
        writer.add_figure(f'cleaned{i}_{datetime.datetime.now()}', plt.gcf())
        writer.flush()
        j += 1
        
        
def plot_to_f(names, tensor_values, savepath,name,title='Correlation histogram'):
    """
    Plots a histogram with the given names on the x-axis and tensor values on the y-axis
    the saves it on a file
    
    Parameters:
    - names (list of str): List of names to display on the x-axis.
    - tensor_values (torch.Tensor): Tensor containing the y-axis values.
    - title (str): Title of the plot (default: 'Correlation histogram').
    """
    if not isinstance(tensor_values, torch.Tensor):
        raise TypeError("tensor_values must be a torch.Tensor")
    if len(names) != tensor_values.numel():
        raise ValueError("The length of names must match the number of elements in tensor_values.")
    
    # Convert tensor to numpy for plotting
    values = tensor_values.numpy()
    
    # Create the histogram
    graph=plt.figure(figsize=(10, 6))
    plt.bar(names, values, color='skyblue', edgecolor='black')
    
    # Add labels and title
    plt.ylabel('Values')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{savepath}/{name}')                    
        
        
        
def plot_stacked_tensor(tensor_to_plot, column_list, f_range, desired_ticks, log_base,path,phase=False, v_max=6,v_min=0,num_plots=10,length_in_s=6):
    """Plots data from stacked_tensor_2d with multiple channels and log-spaced ticks."""
    if phase:
        phase=1
    else:
        phase=0
        
    for i in range(num_plots):
        print('---------------------------')
        print(f'IMAGE {i}')

        # Strain channel is the first channel
        qplt_strain = torch.flipud(tensor_to_plot[i, 0, phase, :, :].detach().cpu())

        # Aux channels are the remaining channels
        aux_images = [torch.flipud(tensor_to_plot[i, j, phase, :, :].detach().cpu()) for j in range(1, tensor_to_plot.shape[1])]

        total_plots = 1 + len(aux_images)
        ncols = min(total_plots, 3)
        nrows = math.ceil(total_plots / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

        if nrows * ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # --- Plot Strain ---
        im = axes[0].imshow(qplt_strain, aspect='auto', vmin=v_min, vmax=v_max)
        axes[0].set_title(column_list[0])  # Use the first column name
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency (Hz)')
        fig.colorbar(im, ax=axes[0])

        set_frequency_ticks(axes[0], f_range, desired_ticks, log_base, qplt_strain.shape[0])
        xlabellist=[i for i in range(qplt_strain.shape[-1]//100+1)]
        xtickslist=list(map(lambda x: x * 100, xlabellist))
        xtickslist[-1]=-1
        #axes[0].set_xticks(xtickslist)
        #axes[0].set_xticklabels(xlabellist)
        axes[0].set_xticks([k* qplt_strain.shape[1]//length_in_s for k in range(length_in_s+1)])
        axes[0].set_xticklabels([k for k in range(length_in_s+1)])
        
        # --- Plot each aux channel ---
        for j, aux_img in enumerate(aux_images):
            ax = axes[j + 1]
            im = ax.imshow(aux_img, aspect='auto', vmin=v_min, vmax=v_max)
            ax.set_title(selected_elements[j+1])  # Use corresponding column name
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            fig.colorbar(im, ax=ax)

            set_frequency_ticks(ax, f_range, desired_ticks, log_base, aux_img.shape[0])
            xlabellist=[i for i in range(aux_img.shape[-1]//100+1)]
            xtickslist=list(map(lambda x: x * 100, xlabellist))
            xtickslist[-1]=-1
            #axes[0].set_xticks(xtickslist)
            #axes[0].set_xticklabels(xlabellist)
            #ax.set_xticks([0, aux_img.shape[1]//6,2*aux_img.shape[1]//6, aux_img.shape[1] // 2,2*aux_img.shape[1]//3,5*aux_img.shape[1]//6,   aux_img.shape[1] - 1])
            #ax.set_xticklabels([0,1,2, 3,4,5, 6]) # adjust time labels based on your time range.
            ax.set_xticks([k* aux_img.shape[1]//length_in_s for k in range(length_in_s+1)])
            ax.set_xticklabels([k for k in range(length_in_s+1)])
            

        for k in range(total_plots, nrows * ncols):
            axes[k].axis('off')

        plt.tight_layout()
        plt.savefig(f'{path}/sample{i}.pdf')
        
        
        
def visualize_dataset(loaded_tensor,run_name,v_max = 15,t_min = 0,f_min = 0,start_aux=1,n_column=3,num_fig=10,strain_channel = 0):
    
    t_max = loaded_tensor.shape[-1]
    f_max = loaded_tensor.shape[-2]
    
    writer = SummaryWriter(run_name)
    
    # For example, if you want to ignore channel 1 and use channels 2 onward as aux:
    aux_channel_indices_all = list(range(start_aux, loaded_tensor.shape[1]))
    # Parameter: how many aux channels you want to plot?
    n_aux_desired = len(aux_channel_indices_all)  # (or set to a lower number, e.g., 2)
    # Select only the desired number of auxiliary channels:
    aux_channel_indices = aux_channel_indices_all[:n_aux_desired]
    
    # Parameter: number of columns for the subplot grid.
    # (Total plots = 1 (strain) + number of aux channels.)
    total_plots = 1 + len(aux_channel_indices)
    ncols = min(total_plots, n_column)  # e.g., up to 3 columns per row; adjust as needed.
    nrows = math.ceil(total_plots / ncols)
    
    for i in range(num_fig):
        print('---------------------------')
        print(f'IMAGE {i}')
    
        # Select and flip images (if desired)
        qplt_strain = torch.flipud(loaded_tensor[i, strain_channel, f_min:f_max, t_min:t_max])
    
        # Create a list to hold aux images.
        aux_images = []
        for idx in aux_channel_indices:
            aux_img = torch.flipud(loaded_tensor[i, idx, f_min:f_max, t_min:t_max])
            aux_images.append(aux_img)

        # Create subplots.
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    
        # Flatten axes array for easier indexing.
        if nrows * ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        
        # Flatten axes array for easier indexing.
        if nrows * ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # --- Plot Strain ---
        im = axes[0].imshow(qplt_strain, aspect='auto', vmin=0, vmax=v_max)
        axes[0].set_title('Strain')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Frequency')
        fig.colorbar(im, ax=axes[0])
    
       # --- Plot each aux channel ---
        for j, aux_img in enumerate(aux_images):
            ax = axes[j + 1]
            im = ax.imshow(aux_img, aspect='auto', vmin=0, vmax=v_max)
            ax.set_title(f'Aux {aux_channel_indices[j]}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            fig.colorbar(im, ax=ax)
    
    # Turn off any extra (unused) subplots.
        for k in range(total_plots, nrows * ncols):
             axes[k].axis('off')
    
        plt.tight_layout()
        writer.add_figure(f'Fig{i}', plt.gcf())
        writer.flush()
    
        plt.close()    
        
def plot_images_gfc(generated_post, batch, channel_means,num_aux_channels=8,num_images=1,v_max=25):
    
    
    
    
    for i in range(num_images):
        print('---------------------------')
        print(f'IMAGE {i}')

        qplt_g = torch.flipud(generated_post[i, 0].detach().cpu() * channel_means[0])
        qplt_r = torch.flipud(batch[i, 0].detach().cpu() * channel_means[0])

        time_extent = generated_post[i, 0].shape[0]
        freq_extent = generated_post[i, 0].shape[1]
        extent = [0, time_extent, 0, freq_extent]

        num_rows_aux = (num_aux_channels + 3) // 4

        fig, axes = plt.subplots(1 + num_rows_aux, 4, figsize=(20, 5 * (1 + num_rows_aux)))

        # Handle the case where there's only one row (including 0 aux channels)
        if 1 + num_rows_aux == 1:  # Only one row
            axes = np.array([axes]) # make axes 2D so that it works with the rest of the code
            axes = axes.reshape(1,4) # reshape it to be a 1x4 array

        im_r = axes[0, 0].imshow(qplt_r, aspect='auto', extent=extent, vmin=0, vmax=v_max)
        axes[0, 0].set_title('Real')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency')
        fig.colorbar(im_r, ax=axes[0, 0])

        im_g = axes[0, 1].imshow(qplt_g, aspect='auto', extent=extent, vmin=0, vmax=v_max)
        axes[0, 1].set_title('Generated')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Frequency')
        fig.colorbar(im_g, ax=axes[0, 1])

        im_diff = axes[0, 2].imshow(torch.abs(qplt_g - qplt_r), aspect='auto', extent=extent, vmin=0, vmax=v_max)
        axes[0, 2].set_title('True - Generated')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Frequency')
        fig.colorbar(im_diff, ax=axes[0, 2])

        axes[0, 3].axis('off')

        aux_channel_index = 1
        row = 1
        col = 0

        for j in range(num_aux_channels):
            qplt_aux = torch.flipud(batch[i, aux_channel_index].detach().cpu() * channel_means[aux_channel_index])
            im_aux = axes[row, col].imshow(qplt_aux, aspect='auto', extent=extent, vmin=0, vmax=v_max)
            axes[row, col].set_title(f'aux{aux_channel_index}')
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel('Frequency')
            fig.colorbar(im_aux, ax=axes[row, col])

            aux_channel_index += 1
            col += 1
            if col == 4:
                col = 0
                row += 1

        for r in range(row, 1 + num_rows_aux):
            for c in range(4):
                axes[r, c].axis('off')

        plt.tight_layout()
        return plt.gcf()
        