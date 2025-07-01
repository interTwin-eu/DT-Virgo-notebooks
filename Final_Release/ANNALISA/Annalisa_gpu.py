import multiprocessing
from tqdm import tqdm
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
import os
import numpy as np
import pandas as pd
import h5py as h5
from os import listdir
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gwpy.signal import filter_design
from . qtransform_gpu import *
from ml4gw.transforms import SpectralDensity,Whiten
from  . Peak_finder_torch import * 
import gc
import torch.nn.functional as F
import torchaudio.transforms as T

from .config import device



from tqdm.auto import tqdm
# Enable tqdm with pandas
tqdm.pandas()


#LOAD AND PREPROCESS DATA

#--------------------------------------------------------------------------------------------------------------------
# Function for loading data from .h5 as pandas df. Add possibility to load data as torch tensor with headers (for channel name). If possible add a label to each row for event id
def construct_dataframe(path, channel_list=None, target_channel='V1:Hrec_hoft_16384Hz', n1_events=None, n2_events=None, n1_channels=None, n2_channels=None, print_=True, sr=False):
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
    - tensor_mode (bool): A boolean indicating whether to load the timeseries as torch tensors

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the HDF5 files.
    """

    # Set default values for event and channel indices if not provided
    if not n1_events:
        n1_events = 0
    if not n2_events:
        n2_events = len(listdir(path))
    
    # Ensure n2_events does not exceed the total number of files in the directory
    if n2_events > len(listdir(path)):
        n2_events = len(listdir(path))
    
    # Get the list of files in the specified directory
    lstdr = listdir(path)[n1_events:n2_events]
    
    # Print the list of files being processed if print_ is True
    #if print_:
        #print(f'LIST DIR: {lstdr}')
    
    # Extract the name of a sample file from the directory
    sample_file = listdir(path)[0]
    
    # Create a list of files to process
    files = [f for f in lstdr]
    
    # Initialize lists to store DataFrame and event data
    df_list = []
    event_data = []
    
    # If channel_list is not provided, get all channels from the first HDF5 file
    if not channel_list:
        n_all_channels = 0
        all_channels = []
        with h5.File(os.path.join(path, sample_file), 'r') as fout:
            event_id = list(fout.keys())[0]
            all_channels = list(fout[event_id])
            n_all_channels = len(list(fout[event_id]))
        
        # Set default values for channel indices if not provided
        if not n1_channels:
            n1_channels = 0
        if not n2_channels:
            n2_channels = n_all_channels
        
        # Ensure n2_channels does not exceed the total number of channels
        if n2_channels > n_all_channels:
            n2_channels = n_all_channels
        
        # Select channels based on provided indices
        channels = all_channels[n1_channels:n2_channels]
    else:
        channels = channel_list
    
    # Remove the target channel from the list of channels
    try:
        channels.remove(target_channel)
    except:
        pass
    
    # Iterate over each file and extract data
    for i, file in enumerate(files):
        if print_:
            print(f"Added {i + 1}/{n2_events - n1_events} files to dataframe", end='\r')
       
        try:
            # Open the HDF5 file
            with h5.File(os.path.join(path, file), 'r') as fout:
                event_id = list(fout.keys())[0]
                dictionary = {'Event ID': event_id}
                event_data.append(event_id)
                
                # Extract data for the target channel
                tmsrs = TimeSeries(fout[event_id][target_channel], dt=1.0 / fout[event_id][target_channel].attrs['sample_rate'],t0=fout[event_id][target_channel].attrs['t0'])

                # Resample the data if required
                if sr:
                    try:
                        tmsrs=tmsrs.resample(sr)
                    except:
                        print('Couldnt resample time series')
                
                dictionary[target_channel] = [tmsrs]
                
                # Extract data for each channel
                for i, channel in enumerate(channels):
                    try:
                       
                        tmsrs = TimeSeries(fout[event_id][channel], dt=1.0 / fout[event_id][channel].attrs['sample_rate'],t0=fout[event_id][target_channel].attrs['t0'])
                        if sr:
                            tmsrs=tmsrs.resample(sr)

                            
                        dictionary[channel] = [tmsrs]
                        
                    except Exception as e:
                        # Handle errors in extracting data
                        tmsrs = np.nan
                        dictionary[channel] = [tmsrs]
                        print(e)
                
                # Convert the dictionary to a DataFrame and append to df_list
                df_list.append(pd.DataFrame(dictionary))
        
        except Exception as e:
            # Handle errors in opening files
            if print_:
                print(f'COULD NOT OPEN {os.path.join(path, file)}')
                print(e)
    
    # Concatenate all DataFrames in df_list into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)
    
    return df
#-------------------------------------------------------------------------------------------------------------------------
def save_dataframe(save_name, out_dir=None,ext='pkl'):
    '''
    Saves dataframe
    
    Parameters:
    - save_name (str): Name of file
    - out_dir (str): out directory where to save file to (default current directory)
    - ext (str): extention of file (default .pkl)
    
    Retruns:
    Nothing
    '''
    
    if out_dir is None:
        out_dir=os.getcwd()
    #save_name='Ts_band_20-60_Hz_whiten_crop_15_channels'
    save_name='Ts_unprocessed_no_resample'
    df.to_pickle(f'{out_dir}/{save_name}.{ext}')
    return
#---------------------------------------------------------------------------------------------------------------------------
def preprocess_timeseries(ts,band_filter=None,whiten=None,duration=None):
    '''
    Process Timeseries by applying band filter, whitening and cropping
    
    Parameters:
    - ts (TimeSeries): time series data to process
    - band_filter (list): frequency window for applying band filter passed as [low_freq,high_freq]. Defalut None, i.e. no band fileter is applied
    - whiten (bool): switch for applying whitening (default None, i.e. no whitening)
    - duration (float): length of output timeseries in seconds. The timeseries is always centered around center of input timeseries. Default None, i.e. no cropping)
    
    Returns:
    Processed Timeseries
    '''
    
    if band_filter:
        low_freq,high_freq=band_filter
        bp = filter_design.bandpass(low_freq, high_freq, ts.sample_rate)
        ts = ts.filter(bp, filtfilt=True)
    if whiten:
        ts=ts.whiten()
    if duration:
        ts=ts.crop(ts.t0.value+(16-duration)/2,ts.t0.value +(16+duration)/2)
    return ts
#--------------------------------------------------------------------------------------------------------------------------------
def find_non_timeseries_entries(df):
    '''
    Checks if the dataframe contains non timeseries entries
    Parameters:
    df (DataFrame): input dataframe of timeseries
    
    Returns:
    non_timeseries_dict (dict): Dictionary containing event id and column name of non timeseries entry

    '''
    # Initialize an empty dictionary to store the results
    non_timeseries_dict = {}
    
    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Get the event id from the first column
        event_id = row.iloc[0]
        # Iterate over the rest of the columns in the row
        for col in df.columns[1:]:
            # Check if the entry is not a TimeSeries
            if not isinstance(row[col], TimeSeries):
                # Add the event id and column name to the dictionary
                if event_id not in non_timeseries_dict:
                    non_timeseries_dict[event_id] = []
                non_timeseries_dict[event_id].append(col)
    
    return non_timeseries_dict
#----------------------------------------------------------------------------------------------------------------------------------
def compute_statistical_dfs(df):
    '''
    Computes multiple dataframes containing stats relative to input dataframe such as maximum values, mean values, mean of absolute valuse, std of mean and abs of mean values.
    
    Parameters:
    df (DataFrame): input dataframe of timeseries
    
    Returns:
    - max_df (DataFrame): daatframe containing maximum value of each timeseries in input df
    - mean_df (DataFrame): dataframe containing mean value of each timeseries in input df
    - mean_abs_df (DataFrame): dataframe containing mean of abs value of each timeseries in input df
    - std_mean_df (DataFrame): dataframe containing std of each timeseries in input df
    - std_mean_abs_df (DataFrame): dataframe containing std of abs value of each timeseries in input df
    '''
    
    # Initialize empty DataFrames with the same shape and index as the input df
    max_df = pd.DataFrame(index=df.index, columns=df.columns)
    mean_df = pd.DataFrame(index=df.index, columns=df.columns)
    mean_abs_df = pd.DataFrame(index=df.index, columns=df.columns)
    std_mean_df = pd.DataFrame(index=df.index, columns=df.columns)
    std_mean_abs_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Iterate over each cell to compute the statistics
    for i in tqdm(df.index):
        for j in df.columns:
            timeseries = df.at[i, j].value
            
            # Calculate the required statistics
            max_df.at[i, j] = np.max(np.abs(timeseries))
            mean_df.at[i, j] = np.mean(timeseries)
            mean_abs_df.at[i, j] = np.mean(np.abs(timeseries))
            std_mean_df.at[i, j] = np.std(timeseries)
            std_mean_abs_df.at[i, j] = np.std(np.abs(timeseries))
    
    return max_df, mean_df, mean_abs_df, std_mean_df, std_mean_abs_df
#-----------------------------------------------------------------------------------------------------------------------------------

def plot_distributions(df,out_dir=None,save_name='distributions',ext='png',num_bins=50):
    '''
    Plots histograms of input df containing stats of timeseries with log spaced bins and saves plot as .png
    
    Parameters:
    - df (DataFrame): input dataframe of stats
    - save_name (str): Name of file (default 'distributions')
    - out_dir (str): out directory where to save file to (default current directory)
    - ext (str): extention of file (default .png)
    - num_bins (int): number of bins for histogram (default 50)
    
    Returns:
    Nothing
    '''
    fig, axes = plt.subplots(4, 4, figsize=(24, 24))
    fig.suptitle('Distributions of Channel Statistics')

    for i, channel in enumerate(df.columns):
        # Get the min and max for log space bins, adding a small offset to avoid log(0)
        min_val = df[channel].min() if df[channel].min() > 0 else 1e-10
        max_val = df[channel].max()
        
        # Create log-spaced bins
        bins = np.logspace(np.log10(min_val), np.log10(max_val), num=num_bins)
        
        # Plot histogram with log-spaced bins
        axes[int(i / 4), i % 4].hist(df[channel], bins=bins, color='skyblue', edgecolor='black')
        axes[int(i / 4), i % 4].set_title(channel)
        axes[int(i / 4), i % 4].set_xscale('log')  # Set x-axis to log scale
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save the figure as .png
    if out_dir is None:
        out_dir = os.getcwd() 
    # Set to current directory 
    plt.savefig(f"{out_dir}/{save_name}.{ext}") 
    # Close the figure to free up memory
    plt.close(fig)
    return
#-----------------------------------------------------------------------------------------------------------------------------------
    
def max_of_channel(arr, channel_index=0):
    '''
    Computes max of a selected channel for an input multichannel array
    
    Parameters:
    - arr (array): array of channels, with each entry being a timeseries
    - channel_index (int): index of channel to consider for max (default 0)
    
    Returns:
    - Max of channel 
    '''
    return np.abs(arr[0]).max()  

#---------------------------------------------------------------------------------------------------------------
def filter_dataframe(df, threshold, func):
    """
    Filters the DataFrame based on a function applied to each row.

    Parameters:
    - df (DataFrame): The input DataFrame to be filtered.
    - threshold (float): The threshold to be used for filtering.
    - func (function): The function to be applied to each row.

    Returns:
    DataFrame: The filtered DataFrame.
    """
    return df[df.apply(lambda x: func(x) < threshold, axis=1)]
#---------------------------------------------------------------------------------
def normalize_data_with_stats(df, df_stats, threshold, func, mode='median', channel_index=0):
    '''
    Filters and normalizes input dataframe based on threshold and desired stats for each channel in the dataframe.
    
    Parameters:
    df (DataFrame): Input dataframe of data to be normalized.
    df_stats (DataFrame): Dataframe containing the statistics of each channel used for normalization.
    threshold (float): The threshold for filtering the data based on the channel statistics.
    func (function): Function to be applied for filtering the dataframe.
    mode (str): The statistical measure to be used for normalization. 
                Options are 'median', 'mean', 'mode', 'std'. Default is 'median'.
    channel_index (int): The index of the channel to use for threshold calculation. Default is 0.
    
    Returns:
    norm_df (DataFrame): The normalized dataframe after filtering and normalization based on the specified statistics.
    '''
    
    if mode == 'median':
        mmm = df_stats.median(axis=0)
    elif mode == 'mean':
        mmm = df_stats.mean(axis=0)
    elif mode == 'mode':
        mmm = df_stats.mode(axis=0).iloc[0]  # mode() returns a DataFrame, take the first row
    elif mode == 'std':
        mmm = df_stats.std(axis=0)
    
    # Define threshold based on desired channel stats
    threshold *= mmm[channel_index]
    
    if threshold is not None:
        # Filter df based on threshold
        df = filter_dataframe(df, threshold, func)
    
    # Normalize filtered dataframe based on stats
    norm_df = df / mmm
    return norm_df
#------------------------------------------------------------------------------------------------------------------------------------------
def normalize_max(entry):
    '''
    Normalize the input entry by dividing each element by the maximum absolute value.

    Parameters:
    entry: array-like
        An array-like structure (e.g., list, numpy array, pandas Series) containing numerical values to be normalized.

    Returns:
    array-like
        The normalized version of `entry`, where each element is divided by the maximum absolute value of the original entry.
        If the maximum is zero, the original entry is returned.
    '''
    mx = np.max(abs(entry))
    if mx != 0:
        return entry / mx
    else:
        return entry
#--------------------------------------------------------------------------------------------------------------------------------------------
def crop_timeseries(ts,t0_shift,duration):
    '''
    Crops time series given a start time and a duration
    
    Parameters:
    - ts (TimeSeries): time series to be cropped
    - t0_shift (float): start time for cropped timeseries
    - duration (float): duration of cropped timeseries
    
    Returns:
    - Cropped timeseries
    '''
    return ts.crop(ts.t0.value+t0_shift,ts.t0.value+t0_shift+duration)
#---------------------------------------------------------------------------------------------------------------------------------------------
def resample_data(entry, sr=200.0):
    '''
    Resamples a time series entry to a specified sampling rate.

    Parameters:
    entry: TimeSeries
        The time series entry to be resampled.
    sr: float, optional
        The desired sampling rate in Hz. Default is 200.0.

    Returns:
    TimeSeries
        The resampled time series entry at the specified sampling rate.
    '''
    return entry.resample(sr)
#--------------------------------------------------------------------------------------------------------------------------------------------
def whiten_(entry):
    '''
    Whitens a TimeSeries entry
    Parameters:
    entry: TimeSeries
        The time series entry to be resampled.
    
    Returns: TimeSeries
        The whitened timeseries
    '''
    norm=abs(entry.value).max()
    
    return ((entry /norm ).whiten())*norm
#---------------------------------------------------------------------------------------------------------------------------------------------------
def band_filter(entry,freq_window):
    '''
    Filters a TimeSeries entry given a frequency window
    Parameters:
    - entry: TimeSeries
        The time series entry to be resampled.
    - freq_window: List
        The frequency window to consider for bandpass filtering
    
    Returns: TimeSeries
        The band filtered timeseries
    '''
    low_freq,high_freq=freq_window
    bp = filter_design.bandpass(low_freq, high_freq, entry.sample_rate)
    
    return entry.filter(bp, filtfilt=True)

#--------------------------------------------------------------------------------------------------------------------------------------------------
def preprocess_timeseries(ts,whiten=None,band_filter=None,duration=None):
    '''
    Preprocess a time series with optional whitening, band-pass filtering, and cropping.

    Parameters:
    ts: TimeSeries object
        The input time series to be preprocessed. Expected to be a numerical data structure representing 
        a signal with methods like `whiten()`, `filter()`, and attributes like `sample_rate` and `t0`.

    whiten: bool, optional
        If True, the function will apply a whitening transformation to the time series. Whitening 
        typically removes correlated noise components to produce a flat frequency spectrum.

    band_filter: tuple, optional
        A tuple of two floats (low_freq, high_freq) specifying the lower and upper cutoff frequencies 
        for band-pass filtering. If provided, the function will apply a band-pass filter to the time series.

    duration: float, optional
        The desired duration (in seconds) to crop the time series. If provided, the function will trim 
        the time series to this duration, centered around the original midpoint.

    Returns:
    TimeSeries object
        The preprocessed time series after applying the specified transformations. The returned time series 
        will have undergone whitening, band-pass filtering, and/or cropping if the corresponding parameters 
        were specified.
    '''
    if whiten:
        #print('Applying Whitening')
        ts= whiten_(ts)
    if band_filter:
        #print('Applying Bandfilter')
        low_freq,high_freq=band_filter
        bp = filter_design.bandpass(low_freq, high_freq, ts.sample_rate)
        ts = ts.filter(bp, filtfilt=True)
    if duration:
        #16 here is hard coded, update with an automatic way to figure out the duration of time series.
        ts=ts.crop(ts.t0.value+(16-duration)/2,ts.t0.value +(16+duration)/2)
    return ts
#------------------------------------------------------------------------------------------------------------------------------------------
#This function needs to be updated with row by row in place modification to df for the sake of memory usage management
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
#------------------------------------------------------------------------------------------------------------------------------------    
# Function to apply in-place operations with generic arguments
def process_element(index, shared_array, shape, func, *args, **kwargs):
    """
    Modifies the element at the specified index in shared_array in place.

    Args:
        index (int): The index of the element to process.
        shared_array (multiprocessing.Array): Shared array representing the DataFrame data.
        shape (tuple): Shape of the original DataFrame.
        func (callable): The function to apply to each element.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    """
    shared_array[index] = func(shared_array[index], *args, **kwargs)

# Function to apply a generic function in parallel with in-place modification
def apply_parallel_inplace_generic(func, df, *args, num_workers=None, **kwargs):
    """
    Applies a function elementwise to a DataFrame in parallel with in-place modification.

    Args:
        func (callable): The function to apply to each element.
        df (pd.DataFrame): The input DataFrame.
        *args: Positional arguments for the function.
        num_workers (int, optional): Number of workers for multiprocessing.
        **kwargs: Keyword arguments for the function.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Flatten the DataFrame to a 1D array
    values = df.values.flatten()
    n = len(values)
    shape = df.shape

    # Create a shared memory array for multiprocessing
    shared_array = multiprocessing.Array('d', values)  # 'd' typecode for double (float64)

    # Define the number of workers (default to the number of CPU cores)
    num_workers = num_workers or multiprocessing.cpu_count()

    # Create a Pool of workers
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Distribute work: apply the function to each element of the shared array
        pool.starmap(
            process_element,
            [(i, shared_array, shape, func, args, kwargs) for i in range(n)]
        )

    # Convert the shared array back to the DataFrame
    result_df = pd.DataFrame(np.array(shared_array).reshape(shape), columns=df.columns)

    return result_df
    
#-----------------------------------------------------------------------------------------------------------------------------------------
def find_nan_indices(df):
    nan_indices = []
    
    # Loop through each entry in the DataFrame
    for i, row in df.iterrows():
        for j, tensor in enumerate(row):
            # Check if the entry is a tensor and contains NaNs
            if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                nan_indices.append((i, j))  # Save the index of the NaN entry
    
    return nan_indices
#-------------------------------------------------------------------------------------------
def count_and_remove_nan_rows(df, nan_indices):
    '''
    Counts and removes rows that have NaN values.
    
    Parameters:
    - df (DataFrame): input dataframe to clean from nans
    - nan_indices (list): list of row indices with nans
    
    Retruns:
    - number of rows with nans (int)
    - df with no nans (DataFame)
    '''
    # Extract unique row indices from nan_indices
    rows_with_nan = set(row for row, _ in nan_indices)
    
    # Count unique rows that contain NaN
    num_rows_with_nan = len(rows_with_nan)
    
    # Drop rows that contain NaN from the DataFrame
    df_cleaned = df.drop(index=rows_with_nan)
    
    return num_rows_with_nan, df_cleaned
#------------------------------------------------
def convert_to_torch(df):
    '''
    Converts input DataFrame into a torch tensor
    
    Parameters:
    - df (DataFrame): input dataframe to be converted to pytorch tensor
    Returns:
    - torch tensor: tensor containg input data
    '''
    return torch.stack([torch.stack([*df.iloc[i]]) for i in range(df.shape[0])])
    
#----------------------------------------------------------------------------------------
class MultiDimBatchDataset(Dataset):
    def __init__(self, data):
        """
        Initialize the dataset.
        :param data: A tensor of shape [num_events, channel_dim, length].
        """
        self.data = data
        self.num_events, self.channel_dim, self.length = data.shape

    def __len__(self):
        """
        Define the dataset length based on the `num_events` dimension.
        """
        return self.num_events

    def __getitem__(self, idx):
        """
        Fetch a single event by index.
        """
        return self.data[idx]


class MultiDimBatchDataLoader:
    def __init__(self, dataset, event_batch_size, channel_batch_size):
        """
        Initialize the multi-dimensional batch data loader.
        :param dataset: The dataset object.
        :param event_batch_size: Batch size for `num_events` dimension.
        :param channel_batch_size: Batch size for `channel_dim` dimension.
        """
        self.dataset = dataset
        self.event_batch_size = event_batch_size
        self.channel_batch_size = channel_batch_size
        self.num_events, self.channel_dim, self.length = dataset.data.shape

    def __iter__(self):
        """
        Create an iterator for the data loader.
        """
        self.event_idx = 0
        self.channel_idx = 0
        return self

    def __next__(self):
        """
        Generate the next batch of data.
        """
        # Check if all events and channels have been processed
        if self.event_idx >= self.num_events:
            raise StopIteration

        # Select a batch of events
        start_event = self.event_idx
        end_event = min(start_event + self.event_batch_size, self.num_events)
        events_batch = self.dataset.data[start_event:end_event]  # Shape: [event_batch, channel_dim, length]

        # Select a batch of channels
        start_channel = self.channel_idx
        end_channel = min(start_channel + self.channel_batch_size, self.channel_dim)
        channel_batch = events_batch[:, start_channel:end_channel, :]  # Shape: [event_batch, channel_batch, length]

        # Update indices for next iteration
        self.channel_idx += self.channel_batch_size
        if self.channel_idx >= self.channel_dim:  # Move to the next event batch when all channels are processed
            self.channel_idx = 0
            self.event_idx += self.event_batch_size

        return channel_batch
    
class NestedMultiDimBatchDataLoader:
    def __init__(self, data, event_batch_size, channel_batch_size):
        """
        Initialize the nested multi-dimensional batch data loader.
        :param data: Input tensor of shape [num_events, num_channels, length].
        :param event_batch_size: Batch size for `num_events` dimension.
        :param channel_batch_size: Batch size for `num_channels` dimension.
        """
        self.data = data
        self.event_batch_size = event_batch_size
        self.channel_batch_size = channel_batch_size
        self.num_events, self.num_channels, self.length = data.shape

    def __iter__(self):
        """
        Create an iterator for the data loader.
        """
        self.event_idx = 0
        return self

    def __next__(self):
        """
        Generate the next parent batch of data.
        """
        # Check if all events have been processed
        if self.event_idx >= self.num_events:
            raise StopIteration

        # Select a batch of events
        start_event = self.event_idx
        end_event = min(start_event + self.event_batch_size, self.num_events)
        parent_batch = self.data[start_event:end_event]  # Shape: [event_batch_size, num_channels, length]

        # Update index for the next iteration
        self.event_idx += self.event_batch_size

        # Return the parent batch for hierarchical processing
        return NestedParentBatch(parent_batch, self.channel_batch_size)


class NestedParentBatch:
    def __init__(self, parent_batch, channel_batch_size):
        """
        Represent a parent batch with child sub-batches.
        :param parent_batch: Tensor of shape [event_batch_size, num_channels, length].
        :param channel_batch_size: Size of each child batch along the `num_channels` dimension.
        """
        self.parent_batch = parent_batch
        self.channel_batch_size = channel_batch_size
        self.num_channels = parent_batch.shape[1]

    def child_batches(self):
        """
        Generate child batches from the parent batch.
        :return: Generator for child batches of shape [event_batch_size, channel_batch_size, length].
        """
        for channel_start in range(0, self.num_channels, self.channel_batch_size):
            channel_end = min(channel_start + self.channel_batch_size, self.num_channels)
            yield self.parent_batch[:, channel_start:channel_end, :]  # Shape: [event_batch_size, channel_batch_size, length]

#-----------------------------------------------------------------------------------------------------------------
def plot_histogram(names, tensor_values, title='Correlation histogram'):
    """
    Plots a histogram with the given names on the x-axis and tensor values on the y-axis.
    
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
    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color='skyblue', edgecolor='black')
    
    # Add labels and title
    plt.ylabel('Values')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    
    # Show the plot
    plt.tight_layout()
    plt.show()
#---------------------------------------------------------------------------------------------------
class Annalisa(nn.Module):
    """
    This class implements a system for computing correlations between Spectrograms. It leverages the Q-transform for time-frequency analysis
    and peak detection to identify salient features in the time series. The core
    functionality revolves around comparing these features between a "strain" time series
    and multiple "auxiliary" time series.

    Args:
        ts_length (int): Length of the time series in samples.
        sample_rate (float): Sampling rate of the time series in Hz.
        device (str, optional): Device to use for computations (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
        threshold (float, optional): SNR^2 threshold for peak detection. Defaults to 20.
        time_window (int, optional): Size of the time window for analysis. Defaults to None.
        time_only_mode (bool, optional): If True, performs comparison only in the time domain, i.e. peaks in normalized energy do not have to occur at same frequency when computing correlations. Defaults to False.
        tolerance_distance (int, optional): Tolerance distance for peak matching. Defaults to 0.
        q (int, optional): Q-value for the Q-transform. Defaults to 12.
        frange (list, optional): Frequency range for the Q-transform [f_min, f_max]. Defaults to [8, 500].
        fres (float, optional): Frequency resolution for the Q-transform. Defaults to 0.5.
        tres (float, optional): Time resolution for the Q-transform. Defaults to 0.1.
        num_t_bins (int, optional): Number of time bins for the Q-transform. Defaults to None.
        num_f_bins (int, optional): Number of frequency bins for the Q-transform. Defaults to None.
        logf (bool, optional): If True, uses logarithmic frequency spacing. Defaults to True.
        qtile_mode (bool, optional): If True, uses quantile-based Q-transform. Defaults to False.
        whiten (bool, optional): If True, applies whitening to the input data. Defaults to False.  # This parameter needs to be updated to None, 'Self','Background', where None corresponds to no whitening,
        # 'Self' computes whitening with respect to the timeseries itself and 'Background'requires the user to pass a psd parameter (needs to be added) in torch.tensor format with dimensions compatible with the input. 
        #We might need two different psd parameters for strain and aux data

    """
    
    def __init__(self, ts_length, sample_rate, device='cpu', threshold=20, time_window=None, time_only_mode=False,
                 tolerance_distance=0, q=12, frange=[10, 50], fres=0.5, tres=0.1, num_t_bins=None, num_f_bins=None,
                 logf=False, qtile_mode=False,whiten=False): #add psd parameter
        #super(Annalisa, self).__init__()
        super().__init__()
        # Set device
        self.device = device

        # Scanner parameters
        self.threshold = threshold
        self.time_window = time_window
        self.time_only_mode = time_only_mode
        self.tolerance_distance = tolerance_distance
        
        # whitening parameters
        self.whiten=whiten
        print(f'{self.whiten=}')
        if whiten:
            self.fftlength = 2
            self.sample_rate=sample_rate

            self.spectral_density = SpectralDensity(
                sample_rate=self.sample_rate,
                fftlength=self.fftlength,
                overlap=None,
                average="median",
            ).to(self.device)
            
            self.fduration=2
            self.whitening = Whiten(
                fduration=self.fduration,
                sample_rate=self.sample_rate,
                highpass=None
            ).to(device)

            

        # QT parameters
        self.length = ts_length
        self.sample_rate = sample_rate
        if whiten:
            self.duration= ts_length / sample_rate - self.fduration
        else:
            self.duration = ts_length / sample_rate
        print(f'{self.duration=}')
        self.q = q
        self.frange = frange
        self.tres = tres
        self.fres = fres
        self.num_t_bins = num_t_bins or int(self.duration / tres)
        self.num_f_bins = num_f_bins or int((frange[1] - frange[0]) / fres)
        self.logf = logf
        self.qtile_mode = qtile_mode

        # Initialize Q-transform
        self.qtransform = SingleQTransform(
            sample_rate=self.sample_rate,
            duration=self.duration,
            q=self.q,
            frange=self.frange,
            spectrogram_shape=(self.num_t_bins, self.num_f_bins),
            logf=self.logf,
            qtiles_mode=self.qtile_mode
        ).to(self.device)
        
        
        #derivatives peak detection

    def forward(self, strain_batch, aux_batch):
        # Compute Q-transform of input data
        if self.whiten:
            #print(f'Before Whiten: {strain_batch.shape=}')
            strain_psd=self.spectral_density(strain_batch.double().to(device))
            #print(f'{strain_psd.shape=}')
            strain_batch = self.whitening(strain_batch.double().to(device), strain_psd)
            #print(f'After Whiten: {strain_batch.shape=}')
            
        qt_strain = self.qtransform(strain_batch.to(self.device))
        #print(f'{qt_strain.shape=}')
        peaks_strain = self.peaks_from_qt_torch(qt_strain, threshold=self.threshold)

        # Correlation coefficients for auxiliary batches
        corr_coeffs = []
        iou_coeffs = []
        for child_aux_batch in aux_batch.child_batches():
            
            if self.whiten:
                aux_psd=self.spectral_density(child_aux_batch.double().to(device))
                child_aux_batch = self.whitening(child_aux_batch.double().to(device), aux_psd)
                #print(f'{aux_psd.shape=}')
                
            qt_aux = self.qtransform(child_aux_batch.to(self.device))
            peaks_aux = self.peaks_from_qt_torch(qt_aux, threshold=self.threshold)
            iou_coeff,corr_coeff = self.compute_ratio(peaks_strain, peaks_aux)
            corr_coeffs.append(corr_coeff)
            iou_coeffs.append(iou_coeff)

        return torch.cat(iou_coeffs, dim=-1).detach().cpu(),torch.cat(corr_coeffs, dim=-1).detach().cpu()

    def peaks_from_qt_torch(self, batch, threshold=25):
        clamped_data = torch.clamp(batch, min=0)
        peaks, _ = find_peaks_torch(clamped_data.flatten(), height=threshold)
        peaks_2d = self.torch_unravel_index(peaks, clamped_data.shape)

        # Create a mask for the detected peaks
        mask = torch.zeros(clamped_data.shape, dtype=torch.bool, device=clamped_data.device)
        mask.index_put_(tuple(peaks_2d.t()), torch.ones(peaks_2d.size(0), dtype=torch.bool, device=clamped_data.device))
        return mask

    def torch_unravel_index(self, indices, shape):
        unraveled_indices = []
        for dim in reversed(shape):
            unraveled_indices.append(indices % dim)
            indices = indices // dim
        return torch.stack(list(reversed(unraveled_indices)), dim=-1)

    def compute_ratio(self, mask1, mask2):
        if self.time_only_mode:
            # Collapse masks along the frequency axis (y-axis)
            mask1 = mask1.any(dim=-2)  # Collapse along frequency axis
            mask2 = mask2.any(dim=-2)  # Collapse along frequency axis

            # Update dimension for summing true elements
            intersection = (mask1 & mask2).sum(dim=-1).float()  # Overlap count per batch (time axis only)
            mask1_count = mask1.sum(dim=-1).float()  # Count in mask1 (time axis only)
            mask2_count = mask2.sum(dim=-1).float()  # Count in mask2 (time axis only)
        else:
            # Compute overlap between masks (full 2D)
            intersection = (mask1 & mask2).sum(dim=(-2, -1)).float()  # Overlap count per batch
            mask1_count = mask1.sum(dim=(-2, -1)).float()  # Count in mask1
            mask2_count = mask2.sum(dim=(-2, -1)).float()  # Count in mask2
            
        #print(f'{mask1_count=}')

        # Calculate Jaccard index (intersection over union)
        union = mask1_count + mask2_count - intersection
        jaccard = intersection / union
        ratio= intersection/mask1_count

        # Handle edge case where intersection and union are zero
        zero_union_mask = (intersection == 0) & (union == 0)
        ratio[zero_union_mask] = 1.0
        jaccard[zero_union_mask] = 1.0


        return torch.nan_to_num(jaccard, nan=0.0),torch.nan_to_num(ratio, nan=0.0)  # Handle cases where union is zero
#-------------------------------------------------------------------------------------------------------------------
class QT_dataset(nn.Module):
    def __init__(self, ts_length, sample_rate, device='cpu', q=12, frange=[8, 500], fres=0.5, tres=0.1, num_t_bins=None, num_f_bins=None,
                 logf=True, qtile_mode=False,whiten=False,psd=None,energy_mode=False,phase_mode=True, window_param = None,tau= 1/2,beta= 8.6):
        super().__init__()
        # Set device
        self.device = device
        
        # whitening parameters
        self.psd=psd
        self.whiten=whiten
        print(f'{self.whiten=}')
        if self.whiten:
            self.fftlength = 2
            self.sample_rate=sample_rate


            self.spectral_density = SpectralDensity(
                sample_rate=self.sample_rate,
                fftlength=self.fftlength,
                overlap=None,
                average="median",
            ).to(self.device)
            
            self.fduration=2
            self.whitening = Whiten(
                fduration=self.fduration,
                sample_rate=self.sample_rate,
                highpass=None
            ).to(device)


            

        # QT parameters
        self.length = ts_length
        self.sample_rate = sample_rate
        if whiten:
            self.duration= ts_length / sample_rate - self.fduration
        else:
            self.duration = ts_length / sample_rate 
        print(f'{self.duration=}')
        self.q = q
        self.frange = frange
        self.tres = tres
        self.fres = fres
        self.num_t_bins = num_t_bins or int(self.duration / tres)
        self.num_f_bins = num_f_bins or int((frange[1] - frange[0]) / fres)
        self.logf = logf
        self.qtile_mode = qtile_mode
        self.energy_mode=energy_mode
        self.phase_mode=phase_mode
        self.tau=tau
        self.beta=beta
        self.window_param=window_param

        # Initialize Q-transform
        self.qtransform = SingleQTransform(
            sample_rate=self.sample_rate,
            duration=self.duration,
            q=self.q,
            frange=self.frange,
            spectrogram_shape=(self.num_f_bins, self.num_t_bins),
            logf=self.logf,
            qtiles_mode=self.qtile_mode,
            energy_mode=self.energy_mode,
            phase_mode=self.phase_mode,
            window_param=self.window_param,
            tau=self.tau,
            beta=self.beta
        ).to(self.device)
        
        
        #derivatives peak detection

    def forward(self, batch):
        # Compute Q-transform of input data
        if self.whiten:
            if self.psd is None:
                #print(f'Before Whiten: {strain_batch.shape=}')
                batch/=torch.max(batch)
                batch_psd=self.spectral_density(batch.double().to(device))
                #print(f'{strain_psd.shape=}')
                batch = self.whitening(batch.double().to(device), batch_psd)
                #print(f'After Whiten: {strain_batch.shape=}')
            else:
                batch/=torch.max(batch)
                #print(f'{strain_psd.shape=}')
                batch = self.whitening(batch.double().to(device), self.psd.double().to(device))
                #this baroque procedure is to ensure that Gracehopper actually frees allocated memeory on gpu. There must be a better way though                
        qt_batch = self.qtransform(batch.to(self.device))
        
        #this baroque procedure is to ensure that Gracehopper actually frees allocated memeory on gpu. There must be a better way though
        processed_batch=qt_batch.detach().cpu()
        del qt_batch
        batch=batch.detach().cpu()
        del batch
        
        torch.cuda.empty_cache()
        gc.collect()


        return processed_batch
#------------------------------------------------------------------------------------------------------------------

#################################################
#################################################


# Vectorized STFT & ISTFT Implementation
###############################################################################
class VectorizedSTFT(nn.Module):
    def __init__(self, 
                 n_fft: int = 1024, 
                 win_length: int = 1024, 
                 hop_length: int = 64, 
                 center: bool = True, 
                 normalized: str = 'energy', 
                 pad_mode: str = "reflect",
                 window: str = 'hann',
                 tukey_alpha: float = 1.0,
                 planck_epsilon: float = 0.5,
                 kaiser_beta: float = 16):
        """
        Vectorized STFT using unfolding.
        
        Args:
            n_fft (int): FFT size.
            win_length (int): Window length (samples).
            hop_length (int): Hop size.
            center (bool): If True, pad the input so frames are centered.
            normalized (str): apply window-power normalization. Possible values 'energy'(mean(window^2)) and 'amplitude (sqrt(mean(window**2)))'
            pad_mode (str): Padding mode.
            window (str): Window type: 'bisquare', 'hann', 'tukey', 'planck-taper', or 'kaiser'. Default 'hann'.
            tukey_alpha (float): Tukey alpha parameter (default 1.0).
            planck_epsilon (float): Planck-taper epsilon parameter (default 0.5).
            kaiser_beta (float): Kaiser beta parameter (default 16).
        """
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.pad_mode = pad_mode
        
        window_type = window.lower() if isinstance(window, str) else 'hann'
        # Compute window in local variable 'win' to avoid duplicate attribute error.
        if window_type == 'tukey':
            win = tukey_window_range(win_length, alpha=2*tukey_alpha, x_min=-1, x_max=1)
        elif window_type == 'planck-taper':
            win = planck_taper_window_range(win_length, epsilon=planck_epsilon, x_min=-1, x_max=1, device='cpu')
        elif window_type == 'kaiser':
            win = kaiser_window_range(win_length, beta=kaiser_beta, x_min=-1, x_max=1, device='cpu')
        elif window_type == 'bisquare':
            win = bisquare_window(win_length, device='cpu')
        else:  # default to hann
            win = torch.hann_window(win_length, periodic=False)
        self.register_buffer("window", win)  # Now register the local variable

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Input [batch, channels, time].
        Returns:
            Tensor: STFT of shape [batch, channels, num_frames, n_fft//2 + 1].
        """
        batch, channels, time_len = x.shape
        if self.center:
            pad = self.n_fft // 2
            x = F.pad(x, (pad, pad), mode=self.pad_mode)
        num_frames = 1 + (x.shape[-1] - self.win_length) // self.hop_length
        # Unfold to obtain overlapping frames: shape [batch, channels, num_frames, win_length]
        x_unfold = x.unfold(dimension=-1, size=self.win_length, step=self.hop_length)
        # Apply window (broadcasted)
        x_windowed = x_unfold * self.window
        # Compute FFT along the window dimension
        stft_result = torch.fft.rfft(x_windowed, n=self.n_fft, dim=-1)
        
        #Apply normalization: because of linearity of FFT we can move this step here 
        if self.normalized=='amplitude':
            norm_factor = self.window.pow(2).sum().sqrt()
        elif self.normalized=='energy':
            norm_factor = torch.mean(self.window**2).item() 
        else:
            norm_factor=1.0
        stft_result = stft_result / norm_factor
        return stft_result
    
class VectorizedISTFT(nn.Module):
    def __init__(self, 
                 n_fft: int = 1024, 
                 win_length: int = 1024, 
                 hop_length: int = 64, 
                 center: bool = True, 
                 normalized: str = 'energy', 
                 window: Optional[torch.Tensor] = None):
        """
        Vectorized inverse STFT using folding.
        
        Args:
            n_fft (int): FFT size.
            win_length (int): Window length.
            hop_length (int): Hop size.
            center (bool): If True, remove padding after reconstruction.
            normalized (bool): If True, apply normalization.
            window (Tensor or None): Synthesis window (default Hann if None).
        """
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        if window is None:
            window = torch.hann_window(win_length, periodic=False)
        if n_fft > win_length:
            window = F.pad(window, (0, n_fft - win_length))
        self.register_buffer("window", window)
    
    def forward(self, stft_matrix: torch.Tensor, length: int):
        """
        Args:
            stft_matrix (Tensor): STFT [batch, channels, num_frames, freq_bins].
            length (int): Desired output length.
        Returns:
            Tensor: Reconstructed signal [batch, channels, time].
        """
        batch, channels, num_frames, freq_bins = stft_matrix.shape
        frames_time = torch.fft.irfft(stft_matrix, n=self.n_fft, dim=-1)
        frames_time = frames_time * self.window
        frames_time = frames_time.reshape(batch * channels, num_frames, self.n_fft)
        frames_time = frames_time.transpose(1, 2)
        out_length = (num_frames - 1) * self.hop_length + self.n_fft
        output = torch.zeros(batch * channels, 1, out_length, device=stft_matrix.device)
        output = F.fold(frames_time, output_size=(1, out_length), kernel_size=(1, self.n_fft), stride=(1, self.hop_length))
        output = output.squeeze(1).squeeze(1)
        if self.center:
            pad = self.n_fft // 2
            output = output[..., pad:-pad]
        output = output.view(batch, channels, -1)
        if self.normalized=='energy':
            norm_factor = self.window.pow(2).sum() / self.hop_length
        elif self.normalized=='amplitude':
            norm_factor = torch.sqrt(self.window.pow(2).sum()) / self.hop_length
        else:
            norm_factor=1
        output = output / norm_factor
        if output.shape[-1] != length:
            output = output[..., :length]
        return output
##############################################################################
###############################################################################
# Windows Implementations
###############################################################################

def planck_taper_window_range(N: int, epsilon: float, x_min: float = -1, x_max: float = 1, scale: float = 1.0, norm=False, device: str = 'cpu') -> torch.Tensor:
    """
    Constructs a Planck-taper window defined over an arbitrary range [x_min, x_max].
    Internally, it maps the input range linearly to the canonical range [-1,1],
    then applies the provided Planck-taper formula.
    
    Args:
        N (int): Window length (number of samples).
        epsilon (float): Taper fraction (0 < epsilon < 0.5).
        x_min (float): Minimum value of the input coordinate.
        x_max (float): Maximum value of the input coordinate.
        norm (str): normalized window's energy to 1
        device (str): Device.
    
    Returns:
        Tensor: A 1D tensor of shape [N] representing the Planck-taper window.
    """
    # Create a coordinate x in [x_min, x_max]
    x = torch.linspace(x_min, x_max, steps=N, device=device, dtype=torch.float64)
    # Apply the scaling to the x coordinate
    x_scaled = scale * x
    # Map scaled x linearly to canonical domain [-1,1]:
    x_canonical = 2 * (x_scaled - x_min) / (x_max - x_min) - 1
    # Map to y in [0,1]
    y = (x_canonical + 1) / 2
    w = torch.ones(N, device=device, dtype=torch.float64)
    # Rising edge: 0 < y < epsilon
    mask_rise = (y > 0) & (y < epsilon)
    if mask_rise.any():
        Z_plus = 2 * epsilon * (1 / (1 + 2 * y[mask_rise] - 1) + 1 / (1 - 2 * epsilon + 2 * y[mask_rise] - 1))
        w[mask_rise] = 1.0 / (torch.exp(Z_plus) + 1.0)
    # Flat region:
    mask_flat = (y >= epsilon) & (y <= 1 - epsilon)
    w[mask_flat] = 1.0
    # Falling edge:
    mask_fall = (y > (1 - epsilon)) & (y < 1)
    if mask_fall.any():
        Z_minus = 2 * epsilon * (1 / (1 - 2 * y[mask_fall] + 1) + 1 / (1 - 2 * epsilon - 2 * y[mask_fall] + 1))
        w[mask_fall] = 1.0 / (torch.exp(Z_minus) + 1.0)
    # Set endpoints to 0
    w[0] = 0.0
    w[-1] = 0.0
    # Zero out-of-bounds regions after scaling
    
    w[(x_scaled < x_min) | (x_scaled > x_max)] = 0.0
    
    if norm:
        w/torch.sqrt(torch.mean(w**2)).item()

    return w

def kaiser_window_range(L: int, beta: float = 8.6, x_min: float = -1, x_max: float = 1, scale: float = 1.0, norm: bool = False, device: str = 'cpu') -> torch.Tensor:
    """
    Returns a Kaiser window of length L defined over an arbitrary range [x_min, x_max].
    (The coordinate mapping is not applied here since torch.kaiser_window returns values independent of x.)

    Args:
        L (int): Window length.
        beta (float): Kaiser beta parameter.
        x_min (float): Minimum coordinate value.
        x_max (float): Maximum coordinate value.
        scale (float): scale the x coordinate.
        norm (bool): normalized window's energy to 1.
        device (str): Device.

    Returns:
        Tensor: A 1D tensor of shape [L] representing the Kaiser window.
    """
    w = torch.kaiser_window(L, beta=beta, periodic=False, device=device, dtype=torch.float64)

    if norm:
        w = w / torch.sqrt(torch.mean(w**2)).item()

    return w

def tukey_window_range(window_length: int, alpha: float = 0.05, x_min: float = 0, x_max: float = 1, scale: float = 1.0, norm: bool = False, device: str = 'cpu') -> torch.Tensor:
    """
    Generates a Tukey window over an arbitrary coordinate range [x_min, x_max].
    The coordinate is mapped linearly to [0,1] and then the Tukey window is computed.

    Args:
        window_length (int): Window length.
        alpha (float): Tukey parameter.
        x_min (float): Minimum coordinate value.
        x_max (float): Maximum coordinate value.
        scale (float): scale the x coordinate.
        norm (bool): normalized window's energy to 1.
        device (str): Device.

    Returns:
        Tensor: A 1D tensor of shape [window_length] representing the Tukey window.
    """
    x = torch.linspace(x_min, x_max, window_length, device=device, dtype=torch.float64)
    x_scaled = scale * x
    x_norm = (x_scaled - x_min) / (x_max - x_min)  # map to [0,1]
    window = torch.ones(window_length, device=device, dtype=torch.float64)
    if alpha == 0:
        return window
    ramp = int(alpha * window_length / 2)
    if ramp == 0:
        return window
    w = torch.linspace(0, 1, ramp, device=device, dtype=torch.float64)
    cosine = 0.5 * (1 + torch.cos(torch.pi * (w - 1)))
    window[:ramp] = cosine
    window[-ramp:] = cosine.flip(0)

    if norm:
        window = window / torch.sqrt(torch.mean(window**2)).item()

    return window

def bisquare_window(L: int, x_min: float = -1, x_max: float = 1, scale: float = 1.0, norm: bool = False, device: str = 'cpu') -> torch.Tensor:
    """
    Compute the bisquare window defined as:
        w(x) = (1 - x^2)^2, with x linearly spaced from -1 to 1.

    Args:
        L (int): Window length.
        x_min (float): Minimum coordinate value.
        x_max (float): Maximum coordinate value.
        scale (float): scale the x coordinate.
        norm (bool): normalized window's energy to 1.
        device (str): Device.

    Returns:
        Tensor: A 1D tensor of shape [L] representing the bisquare window.
    """
    x = torch.linspace(x_min, x_max, steps=L, device=device, dtype=torch.float64)
    x_scaled = scale * x
    x_norm = 2 * (x_scaled - x_min) / (x_max - x_min) -1
    w = (1 - x_norm**2)**2

    if norm:
        w = w / torch.sqrt(torch.mean(w**2)).item()

    return w

def hann_window_range(L: int, x_min: float = 0, x_max: float = 1, scale: float = 1.0, norm: bool = False, device: str = 'cpu') -> torch.Tensor:
    """
    Generates a Hann window over an arbitrary coordinate range [x_min, x_max].
    The coordinate is mapped linearly to [0,1] and then the Hann window is computed.

    Args:
        L (int): Window length.
        x_min (float): Minimum coordinate value.
        x_max (float): Maximum coordinate value.
        scale (float): scale the x coordinate.
        norm (bool): normalized window's energy to 1.
        device (str): Device.

    Returns:
        Tensor: A 1D tensor of shape [L] representing the Hann window.
    """
    x = torch.linspace(x_min, x_max, L, device=device, dtype=torch.float64)
    x_scaled = scale * x
    x_norm = (x_scaled - x_min) / (x_max - x_min)  # map to [0,1]
    window = torch.hann_window(L, periodic=False, device=device, dtype=torch.float64)

    if norm:
        window = window / torch.sqrt(torch.mean(window**2)).item()

    return window
    
    
    
    
###############################################################################    
# WHITENING
###############################################################################
class STFTWhiten(nn.Module):
    def __init__(self, 
                 # Parameters for PSD estimation:
                 psd_nfft: int = 2048, 
                 psd_win_length: int = 2048, 
                 psd_hop_length: Optional[int] = None,
                 # Parameters for whitening STFT:
                 stft_nfft: int = 1024, 
                 stft_win_length: int = 1024, 
                 stft_hop_length: int = 32,
                 sample_rate: float = 4096.0,
                 center: bool = True, 
                 normalized: str = 'energy',
                 average_type: str = 'mean',   # 'mean', 'median', 'moving_mean', 'moving_median'
                 normalization: str = 'ml4gw',   # 'ml4gw', 'nperseg', 'window_sum', 'fftlength', None
                 moving_avg_window_ratio: float = 0.2,  
                 epsilon: float = 0,              
                 # Border mitigation options:
                 exclude_border: bool = False,
                 border_fraction: float = 0.01,
                 border_mitigation: bool = True,   # Set to True to enable border mitigation.
                 tapering: bool = True, 
                 zero_phase_padding: bool = False,
                 detrend: bool = True,
                 stft_frame_weighting: bool = False,
                 pad_mode: str = "reflect",
                 # Synthesis window compensation option:
                 design_synth_window: bool = True,
                 # New window options for PSD and STFT:
                 truncation_window_type: str = 'hann',  # Options: 'bisquare','hann','tukey','planck-taper','kaiser'
                 truncation_tukey_alpha: float = 1.0,
                 truncation_planck_epsilon: float = 0.5,
                 truncation_kaiser_beta: float = 16,
                 truncation_window_size=2, # in seconds
                 truncation_window_scale=1.0,
                 psd_window_type: str = 'hann',  # Options: 'bisquare','hann','tukey','planck-taper','kaiser'
                 psd_tukey_alpha: float = 1.0,
                 psd_planck_epsilon: float = 0.5,
                 psd_kaiser_beta: float = 16,
                 stft_window_type: str = 'hann',  # Options: same as above
                 stft_tukey_alpha: float = 1.0,
                 stft_planck_epsilon: float = 0.5,
                 stft_kaiser_beta: float = 16,
                 convolve_method='overlapsave',
                 eps_thresh: float = 0,#50.0,  # frequency threshold in Hz
                 eps_slope: float = 5.0,    # transition width in Hz
                 eps_max: float = 1e-21, 
                 # Set device
                 device: str = 'cpu'
                ):
        """
        STFT-based whitening module with separate parameters for PSD estimation and whitening,
        plus advanced border mitigation strategies.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.center = center
        self.normalized = normalized
        self.average_type = average_type
        self.moving_avg_window_ratio = moving_avg_window_ratio
        self.epsilon = epsilon
        self.normalization = normalization
        self.exclude_border = exclude_border
        self.border_fraction = border_fraction
        self.border_mitigation = border_mitigation
        self.tapering = tapering
        self.zero_phase_padding = zero_phase_padding
        self.detrend = detrend
        self.stft_frame_weighting = stft_frame_weighting
        self.pad_mode = pad_mode
        self.convolve_method = convolve_method
        self.truncation_window_type = truncation_window_type
        self.truncation_window_size = truncation_window_size
        self.eps_thresh = eps_thresh
        self.eps_slope = eps_slope
        self.eps_max = eps_max
        self.psd_tail=None

        # -------------------------------
        # Set up window for whitening STFT:
        if stft_win_length is None:
            stft_win_length = stft_nfft
        analysis_window = self._get_window(
            win_length=stft_win_length,
            window_type=stft_window_type,
            tukey_alpha=stft_tukey_alpha,
            planck_epsilon=stft_planck_epsilon,
            kaiser_beta=stft_kaiser_beta,
            device=device
        )
        if stft_nfft > stft_win_length:
            analysis_window = F.pad(analysis_window, (0, stft_nfft - stft_win_length))
        self.register_buffer('analysis_window', analysis_window)
        
        if design_synth_window:
            synth_window = self._compute_synthesis_window(analysis_window, stft_hop_length)
        else:
            synth_window = analysis_window.clone()
        if stft_nfft > stft_win_length:
            synth_window = F.pad(synth_window, (0, stft_nfft - stft_win_length))
        self.register_buffer('synthesis_window', synth_window)
        
        # -------------------------------
        # Set up window for PSD estimation:
        if psd_win_length is None:
            psd_win_length = psd_nfft
        psd_window = self._get_window(
            win_length=psd_win_length,
            window_type=psd_window_type,
            tukey_alpha=psd_tukey_alpha,
            planck_epsilon=psd_planck_epsilon,
            kaiser_beta=psd_kaiser_beta,
            device=device
        )
        if psd_nfft > psd_win_length:
            psd_window = F.pad(psd_window, (0, psd_nfft - psd_win_length))
        self.register_buffer('psd_window', psd_window)

        # -------------------------------
        # Set up window for PSD truncation:
        truncation_win_length = int(truncation_window_size * sample_rate)
        truncation_window = self._get_window(
            win_length=truncation_win_length,
            window_type=truncation_window_type,
            tukey_alpha=truncation_tukey_alpha,
            planck_epsilon=truncation_planck_epsilon,
            kaiser_beta=truncation_kaiser_beta,
            device=device,
            scale=truncation_window_scale
        )
        self.register_buffer('truncation_window', truncation_window)
        
        # -------------------------------
        # Instantiate vectorized STFT and ISTFT modules.
        self.psd_stft = VectorizedSTFT(n_fft=psd_nfft, win_length=psd_win_length, 
                                       hop_length=psd_hop_length if psd_hop_length is not None else psd_win_length // 2, 
                                       center=center, normalized=normalized, pad_mode=pad_mode)
        
        self.signal_stft = VectorizedSTFT(n_fft=stft_nfft, win_length=stft_win_length, 
                                          hop_length=stft_hop_length, 
                                          center=center, normalized=normalized, pad_mode=pad_mode)
        
        self.signal_istft = VectorizedISTFT(n_fft=stft_nfft, win_length=stft_win_length, 
                                            hop_length=stft_hop_length, center=center, normalized=normalized,
                                            window=synth_window)


    def _frequency_dependent_epsilon(self, N_fft: int) -> torch.Tensor:
        """
        Create a 1D tensor of epsilon values for frequencies corresponding
        to a real FFT of length N_fft, with a transition starting at eps_thresh.
        """
        # Compute frequencies for rFFT:
        # f_k = k * sample_rate / N_fft for k=0...N_fft//2
        freqs = torch.linspace(0, self.sample_rate / 2, steps=N_fft // 2 + 1, device=self.synthesis_window.device)
        # Compute epsilon values using a sigmoid transition:
        # epsilon(f) = eps_max / (1 + exp(-(f - eps_thresh) / eps_slope))
        eps_values = self.eps_max / (1.0 + torch.exp(-(freqs - self.eps_thresh) / self.eps_slope))
        return eps_values  # shape [N_fft//2+1]
    
    def _get_window(self, win_length, window_type='hann', x_min=-1, x_max=1, scale=1.0,
                    tukey_alpha: float = 1.0, planck_epsilon: float = 0.5, kaiser_beta: float = 16,
                    device: str = 'cpu'):
        """
        Helper method to create a window.
        """
        window_type = window_type.lower() if isinstance(window_type, str) else 'hann'
        if window_type == 'tukey':
            return tukey_window_range(win_length, alpha=2 * tukey_alpha, x_min=x_min, x_max=x_max, scale=scale)
        elif window_type == 'planck-taper':
            return planck_taper_window_range(win_length, epsilon=planck_epsilon, x_min=x_min, x_max=x_max, device=device, scale=scale)
        elif window_type == 'kaiser':
            return kaiser_window_range(win_length, beta=kaiser_beta, x_min=x_min, x_max=x_max, device=device, scale=scale)
        elif window_type == 'bisquare':
            return bisquare_window(win_length, device=device, scale=scale)
        return hann_window_range(win_length, device=device, scale=scale)
    
    def _compute_synthesis_window(self, analysis_window, hop_length):
        """
        Compute a synthesis window that compensates for non-COLA behavior.
        """
        L = analysis_window.shape[0]
        n = torch.arange(L, dtype=torch.long, device=analysis_window.device)
        c = torch.zeros(L, dtype=analysis_window.dtype, device=analysis_window.device)
        m_min = -((L - 1) // hop_length)
        m_max = (L - 1) // hop_length
        for m in range(m_min, m_max + 1):
            idx = n - m * hop_length
            mask = (idx >= 0) & (idx < L)
            if mask.any():
                c[mask] += analysis_window[idx[mask]]
        return analysis_window / (c.clamp(min=1e-8))
    
    def _compute_normalization_scale(self, window):
        """
        Compute the normalization scale factor based on the chosen method.
        """
        if self.normalization == 'energy':
            self.scale = 1.0 / torch.sqrt(torch.mean(window**2)).item()
        if self.normalization == 'ml4gw':
            self.scale = 1.0 / (self.sample_rate * (window ** 2).sum())
        elif self.normalization == 'nperseg':
            self.scale = 1.0 / window.numel()
        elif self.normalization == 'window_sum':
            self.scale = 1.0 / window.sum()
        elif self.normalization == 'fftlength':
            self.scale = 1.0 / self.signal_stft.n_fft
        else:
            self.scale = 1.0
            
    def _detrend_data(self, x):
        # Detrend (subtract mean)
        return x - x.mean(dim=-1, keepdim=True)
        
    def _border_mitigation(self, x):
        """
        Apply border mitigation to the raw data.
        2. Pad the signal by a chosen amount (here, half the PSD STFT n_fft).
        3. Taper the edges using a selectable window.
        Returns the processed signal and the pad amount.
        """
        # 2. Pad the signal (only once here)
        self.pad_amount = int(x.shape[-1] *self.border_fraction)
        
        #apply small taper at tails in order to avoid non differentiability introduced by reflection or discontinuity introduced by zero padding (and hence high frequency artefacts)
        taper_amount=int(1*self.sample_rate)
        taper = self._get_window(2*taper_amount, window_type='planck-taper', device=x.device)
        x[...,:taper_amount] *= taper[:taper_amount]
        x[...,-taper_amount:] *= taper[taper_amount:]
        
        x = F.pad(x, (self.pad_amount, self.pad_amount), mode=self.pad_mode)
        # 3. Taper the edges: using _get_window to choose the window function.
        taper = self._get_window(2*self.pad_amount, window_type='hann', device=x.device)
        x[...,:self.pad_amount] *= taper[:self.pad_amount]
        x[...,-self.pad_amount:] *= taper[self.pad_amount:]
        return x, self.pad_amount

    def _psd_truncation(self, whitening_filter, truncation_window):
        """
        Perform time-domain truncation on the whitening filter.
        """
        fbins = whitening_filter.shape[-1]
        N_t = 2 * (fbins - 1)
        h_time = torch.fft.irfft(whitening_filter.squeeze(2), n=N_t, dim=-1)
        # Roll to center the impulse using half the truncation window length minus 1.
        h_rolled = torch.roll(h_time, shifts=int(truncation_window.shape[-1] / 2 - 1), dims=-1)
        h_windowed = h_rolled[..., :int(truncation_window.shape[-1])] * truncation_window / (torch.sqrt(torch.mean(truncation_window**2)).item())
        # Roll back so that the impulse peak is at t=0 (for FFT-based filter design).
        h_wind_roll = torch.roll(h_windowed, shifts=-int(truncation_window.shape[-1] / 2 - 1), dims=-1)
        H_truncated = torch.fft.rfft(h_wind_roll, n=N_t, norm="forward", dim=-1)
        inv_psd = H_truncated * H_truncated.conj()
        new_psd = 1.0 / (inv_psd.abs() + self.epsilon)
        return 1.0 / torch.sqrt(new_psd)
        
    def _truncate_transfer_torch(self, transfer, ncorner=0):
        """
        Mimic gwpy's truncate_transfer in torch.
        (Documentation unchanged.)
        """
        nsamp = transfer.shape[-1]
        out = transfer.clone()
        if ncorner > 0:
            out[..., :ncorner] = 0
        taper = planck_taper_window_range(nsamp - ncorner, epsilon=ncorner / nsamp, x_min=-1, x_max=1, scale=1.0, device=transfer.device)
        out[..., ncorner:] *= taper
        return out

    def _psd_truncation_time(self, whitening_filter, truncation_window):
        """
        Perform time-domain truncation on the whitening filter.
        """
        fbins = whitening_filter.shape[-1]
        N_t = 2 * (fbins - 1)
        h_time = torch.fft.irfft(whitening_filter.squeeze(2), n=N_t, dim=-1)
        # For time-domain convolution, center the impulse without rolling it back.
        h_rolled = torch.roll(h_time, shifts=int(truncation_window.shape[-1] / 2 - 1), dims=-1)
        h_windowed = h_rolled[..., :int(truncation_window.shape[-1])] * truncation_window / (torch.sqrt(torch.mean(truncation_window**2)).item())
        return h_windowed

    def overlap_save_convolve(self, x: torch.Tensor, fir: torch.Tensor, window: str = 'hann') -> torch.Tensor:
        """
        Args:
            x: Input signal [batch, channels, time]
            fir: FIR filter [batch, channels, filter_length]
            window: Window type ('hann', 'tukey', 'planck-taper', etc.)
        
        Returns:
            y: Convolved signal [batch, channels, time]
        """
        fir = fir.squeeze(2)
        batch, channels, time_len = x.shape
        filter_length = fir.shape[-1]
        pad = int(torch.ceil(torch.tensor(filter_length / 2.0)).item())
        nfft = min(8 * filter_length, time_len)
        
        # In overlap-save, we assume that border mitigation already padded x.
        # 1. Apply window to input edges.
        window_tensor = self._get_window( 2 * pad,window, device=x.device)
        x_tapered = x.clone()
        x_tapered[..., :pad] *= window_tensor[:pad]
        x_tapered[..., -pad:] *= window_tensor[-pad:]
        
        # 2. Overlap-save logic.
        if nfft >= time_len / 2:
            #Single convolution
            y = T.FFTConvolve(mode='same')(x_tapered, fir)
        else:
            #Overlap save convolution
            nstep = nfft - 2 * pad
            y = torch.zeros_like(x)
            for k in range(0, time_len, nstep):
                start = max(k - pad, 0)
                end = min(k + nstep + pad, time_len)
                chunk = x_tapered[..., start:end]
                if end - start < nfft:
                    chunk = F.pad(chunk, (0, nfft - (end - start)))
                conv_chunk = T.FFTConvolve(mode='same')(chunk, fir)
                valid_start = pad if start > 0 else 0
                valid_end = -pad if end < time_len else None
                y[..., k:min(k + nstep, time_len)] = conv_chunk[..., valid_start:valid_end]
        # No extra cropping here since border mitigation is done only once.
        return y
    
    def compute_psd(self, x):
        """
        Compute the PSD using the vectorized STFT for PSD estimation.
        (Documentation unchanged.)
        """
        stft = self.psd_stft(x)
        psd = torch.abs(stft) ** 2
        psd /= self.sample_rate**2  # to match gwpy
        if self.exclude_border:
            tb = psd.shape[2]
            num_exclude = int(self.border_fraction * tb)
            psd = psd[..., num_exclude:tb - num_exclude, :]
            
        self.psd_tail= psd.mean(dim=2, keepdim=True)   #compute mean psd as more reliable for tails
        
        if self.average_type is None:
            psd_est = psd.mean(dim=2, keepdim=True)
        elif self.average_type == 'mean':
            psd_est = psd.mean(dim=2, keepdim=True)
        elif self.average_type == 'median':
            psd_est = psd.median(dim=2, keepdim=True).values
            
        elif self.average_type in ['moving_mean', 'moving_median']:
            time_bins = psd.shape[2]
            window_size = max(1, int(self.moving_avg_window_ratio * time_bins))
            psd_unfold = psd.unfold(dimension=2, size=window_size, step=1)
            if self.average_type == 'moving_median':
                psd_est = psd_unfold.median(dim=-1).values.unsqueeze(2)
            else:
                psd_est = psd_unfold.mean(dim=-1).unsqueeze(2)
        else:
            raise ValueError("Unsupported average_type.")
        psd_est = psd_est
        return psd_est
    
    def _remove_borders(self, x, pad_amount):
        # Crop out the padded region.
        x = x[..., pad_amount:-pad_amount]

        # Define a fraction of the truncation window size to use as the final taper length.
        # For instance, 10% of the truncation window length.
        final_taper_fraction = 0.1  
        trunc_win_len = self.truncation_window.shape[-1]  # Number of samples in the truncation window.
        taper_length = int(trunc_win_len * final_taper_fraction)

        # Only apply taper if the final taper length is smaller than the signal length.
        if 2 * taper_length < x.shape[-1]:
            # Create a Planck-taper window of total length 2 * taper_length.
            # (Assuming planck_taper_window_range returns a 1D tensor of the specified length.)
            final_taper = planck_taper_window_range(2 * taper_length, epsilon=0.5, x_min=-1, x_max=1, scale=1.0, device=x.device)
            # Apply the taper to the beginning.
            x[..., :taper_length] *= final_taper[:taper_length]
            # Apply the taper to the end.
            x[..., -taper_length:] *= final_taper[-taper_length:]
        return x
    
    def _blend_filters(self, wf_nom, wf_eps):
        """
        Blend two whitening filters (nominal and high-epsilon) in the time domain.
        wf_nom and wf_eps are assumed to have shape [1, 1, T], where T is the filter length.
        We define a blending mask that is 1 in the central region and 0 at the borders,
        with a smooth transition over a specified number of samples.
        """
        T = wf_nom.shape[-1]
        # Define the blending region as a fraction of T. For instance, use 50% of the filter length as the blend width.
        blend_width = int(0.5 * T)
        # Create a Hann window of length 2*blend_width.
        #if 2 * blend_width > T:
            #blend_width = T // 2
        hann_win = torch.hann_window(2 * blend_width, periodic=True, device=wf_nom.device)
        # Construct a blending mask of length T.
        blend_mask = torch.ones(T, device=wf_nom.device)

        blend_mask[:blend_width] =0
        blend_mask[-blend_width:] =0 
        blend_mask = blend_mask.unsqueeze(0).unsqueeze(0)  # shape [1, 1, T]
        # Blend the filters:
        wf_blended = blend_mask * wf_nom + (1 - blend_mask) * wf_eps
        return wf_blended
    
    def forward(self, x, background_psd=None):
        """
        Apply whitening to the input signal.
        (Documentation unchanged.)
        """
        batch, channels, time_len = x.shape

        # Detrend the data.
        x = self._detrend_data(x)

        # Compute the PSD on the detrended data.
        if background_psd is not None:
            psd_est = background_psd.unsqueeze(2) if background_psd.ndim == 3 else background_psd
        else:
            psd_est = self.compute_psd(x)
            whitening_filter_tail = 1.0 / (torch.sqrt(self.psd_tail))

            x_roll=torch.roll(x, shifts=int(x.shape[-1] / 2 - 1), dims=-1)
            psd_est_roll=self.compute_psd(x_roll)

        # Compute nominal and high-ε whitening filters.
        whitening_filter_nom = 1.0 / torch.sqrt(psd_est)
        #whitening_filter_eps = 1.0 / (torch.sqrt(psd_est) + self.epsilon)
        


        # Compute whitening filters using frequency-dependent epsilon.
        whitening_filter_eps = 1.0 / (torch.sqrt(psd_est) + self.epsilon)
        

        if self.convolve_method == 'stft':
            # Process both filters via your time-domain truncation.
            wf_nom = self._psd_truncation(whitening_filter_nom, self.truncation_window.unsqueeze(0).unsqueeze(0))
            wf_eps = self._psd_truncation(whitening_filter_eps, self.truncation_window.unsqueeze(0).unsqueeze(0))
            wf_tail = self._psd_truncation(whitening_filter_tail, self.truncation_window.unsqueeze(0).unsqueeze(0))
            
            wf_nom = wf_nom.unsqueeze(2)
            wf_eps = wf_eps.unsqueeze(2)
            wf_tail = wf_tail.unsqueeze(2)
            

            # Apply border mitigation (pad & taper) once.
            if self.border_mitigation:
                x, pad_amount = self._border_mitigation(x)
                batch, channels, time_len = x.shape
            else:
                pad_amount = 0
                

            stft_signal = self.signal_stft(x)


            # Ensure the whitening filters match the STFT time-dimension.
            if psd_est.shape[-1] != stft_signal.shape[-1]:
                wf_nom = F.interpolate(wf_nom, size=(1, stft_signal.shape[-1]), mode='bicubic', align_corners=False)
                wf_eps = F.interpolate(wf_eps, size=(1, stft_signal.shape[-1]), mode='bicubic', align_corners=False)

            # --- Construct Blending Mask Based on Border Region ---
            total_bins = stft_signal.shape[-1]
            # Determine border region based on pad_amount and half the filter length in samples.
            # Assume the filter length (in seconds) is truncation_window_size.
            filter_samples = int(self.truncation_window_size * self.sample_rate)  # e.g., 2s * 4096 Hz = 8192 samples.
            # Half filter length in samples.
            half_filter = filter_samples // 2  
            # The border region in samples is then:
            border_samples = pad_amount + half_filter  
            # Convert border_samples to a fraction of total STFT time bins:
            # (This conversion depends on how time in STFT corresponds to original samples;
            # often, time bins = (signal_length / hop_length). For simplicity, assume a linear mapping.)
            # Here we simply clamp border_samples to be at most half the total bins.
            border_bins = min(total_bins // 2, border_samples)

            # Create a blending mask that ramps from 0 to 1 over border_bins at each end.
            blend_mask = torch.ones(total_bins, device=stft_signal.device)

            blend_mask[:border_bins]=0
            blend_mask[-border_bins:]=0
            
            blend_mask = blend_mask.unsqueeze(0).unsqueeze(0)  # shape [1, 1, total_bins]

            # --- Apply Filters and Blend ---
            stft_whitened_nom = stft_signal * wf_nom
            stft_whitened_eps = stft_signal * wf_eps
            stft_whitened_tail= stft_signal * wf_tail
            stft_whitened = blend_mask * stft_whitened_nom + (1 - blend_mask) * stft_whitened_eps
            stft_whitened_nt = blend_mask * stft_whitened_nom + (1 - blend_mask) * stft_whitened_tail
            
            
            print(f'{time_len=}')

            x_whitened = self.signal_istft(stft_whitened_nt, length=time_len)

            # Remove border mitigation padding.
            if self.border_mitigation:
                x_whitened = x_whitened[..., pad_amount:-pad_amount]

        else:
            # Overlap-save branch: similar idea applies.
            wf_nom = self._psd_truncation_time(whitening_filter_nom, self.truncation_window.unsqueeze(0).unsqueeze(0))
            wf_eps = self._psd_truncation_time(whitening_filter_eps, self.truncation_window.unsqueeze(0).unsqueeze(0))
            wf_nom = wf_nom.unsqueeze(2)
            wf_eps = wf_eps.unsqueeze(2)
    
            if self.border_mitigation:
                x, pad_amount = self._border_mitigation(x)

            else:
                pad_amount = 0
                
            wf_blended = self._blend_filters(wf_nom, wf_eps)
            # Then use the blended filter in the overlap-save convolution.
            x_roll=torch.roll(x, shifts=int(x.shape[-1] / 2 - 1), dims=-1)
            
            x_w_rolled= self.overlap_save_convolve(x_roll, wf_blended, window=self.truncation_window_type)
            x_w_unrolled=torch.roll(x_w_rolled, shifts=- int(x.shape[-1] / 2 - 1), dims=-1)
            
            x_whitened = self.overlap_save_convolve(x, wf_blended, window=self.truncation_window_type)
            # Remove the border mitigation padding.
            if self.border_mitigation:
                
                x_whitened = self._remove_borders(x_whitened, pad_amount)
                x_w_unrolled= self._remove_borders(x_w_unrolled, pad_amount)
                
            

        return x_whitened.real #x_whitened.real



####################################################

class QT_dataset_custom(nn.Module):
    def __init__(self, ts_length, sample_rate, device='cpu', q=12, frange=[8, 500], fres=0.5, tres=0.1, num_t_bins=None, num_f_bins=None,
                 logf=True, qtile_mode=False,whiten=False,whiteconf=None,psd=None,energy_mode=False,phase_mode=True, window_param = None,tau= 1/2,beta= 8.6):
        super().__init__()
        # Set device
        self.device = device
        
        # whitening parameters
        self.psd=psd
        self.whiten=whiten
        self.whiteconf=whiteconf
        self.sample_rate = sample_rate
        print(f'{self.whiten=}')
        # whitening parameters
        self.psd=psd
        self.whiten=whiten
        print(f'{self.whiten=}')
        if whiten:
            self.fduration=0
            self.whitening = STFTWhiten(
                psd_nfft=self.whiteconf['psd_nfft'], 
                psd_win_length=self.whiteconf['psd_win_length'], 
                psd_hop_length=self.whiteconf['psd_hop_length'],
                # Parameters for whitening STFT:
                stft_nfft=self.whiteconf['stft_nfft'], #1024
                stft_win_length=self.whiteconf['stft_win_length'], #1024
                stft_hop_length=self.whiteconf['stft_hop_length'],  #1024
                stft_window_type=self.whiteconf['stft_window_type'], 
                sample_rate=self.sample_rate,
                center=self.whiteconf['center'], 
                normalized=self.whiteconf['normalized'],
                average_type=self.whiteconf['average_type'],   # Options: None, 'mean', 'median', 'moving_mean', 'moving_median'
                normalization=self.whiteconf['normalization'],   # Options: 'ml4gw', 'nperseg', 'window_sum', 'fftlength', None
                moving_avg_window_ratio=self.whiteconf['moving_avg_window_ratio'],  
                epsilon=self.whiteconf['epsilon'],#1.0e-21,#1.0e-12,              
                # Border mitigation options:
                exclude_border=self.whiteconf['exclude_border'],       # Exclude border frames in PSD estimation.
                border_fraction=self.whiteconf['border_fraction'],       # Fraction of frames to exclude at each end.
                border_mitigation=self.whiteconf['border_mitigation'],    
                pad_mode=self.whiteconf['pad_mode'],
                # Synthesis window compensation option:
                design_synth_window=self.whiteconf['design_synth_window'],
                # New window options for PSD and STFT:
                psd_window_type=self.whiteconf['psd_window_type'],
                psd_planck_epsilon=self.whiteconf['psd_planck_epsilon'],
                psd_kaiser_beta=self.whiteconf['psd_kaiser_beta'],
                truncation_window_type=self.whiteconf['truncation_window_type'],
                truncation_planck_epsilon=self.whiteconf['truncation_planck_epsilon'],
                truncation_kaiser_beta=self.whiteconf['truncation_kaiser_beta'],
                detrend=self.whiteconf['detrend'],
                truncation_window_size=self.whiteconf['truncation_window_size'],
                convolve_method=self.whiteconf['convolve_method']  #'overlapsave' 'stft'
            ).to(device)
            # @Alessio parameters are hard coded, you should allow the user to change them


            
        # QT parameters
        self.length = ts_length
        #self.sample_rate = sample_rate
        if whiten:
            self.duration= ts_length / sample_rate - self.fduration
        else:
            self.duration = ts_length / sample_rate 
        print(f'{self.duration=}')
        self.q = q
        self.frange = frange
        self.tres = tres
        self.fres = fres
        self.num_t_bins = num_t_bins or int(self.duration / tres)
        self.num_f_bins = num_f_bins or int((frange[1] - frange[0]) / fres)
        self.logf = logf
        self.qtile_mode = qtile_mode
        self.energy_mode=energy_mode
        self.phase_mode=phase_mode
        self.tau=tau
        self.beta=beta
        self.window_param=window_param

        # Initialize Q-transform
        self.qtransform = SingleQTransform(
            sample_rate=self.sample_rate,
            duration=self.duration,
            q=self.q,
            frange=self.frange,
            spectrogram_shape=(self.num_f_bins, self.num_t_bins),
            logf=self.logf,
            qtiles_mode=self.qtile_mode,
            energy_mode=self.energy_mode,
            phase_mode=self.phase_mode,
            window_param=self.window_param,
            tau=self.tau,
            beta=self.beta
        ).to(self.device)
        
        
    def forward(self, batch):
        # Compute Q-transform of input data
        if self.whiten:
            if self.psd is None:
                print(f'Before whitening: {batch.shape=}')
                batch = self.whitening(batch.to(device))
                print(f'After whitening: {batch.shape=}')
            else:
                batch = self.whitening(batch.to(device), self.psd.double().to(device))             
        qt_batch = self.qtransform(batch.to(self.device))
        
        #this baroque procedure is to ensure that Gracehopper actually frees allocated memeory on gpu. There must be a better way though
        processed_batch=qt_batch.detach().cpu()
        del qt_batch
        batch=batch.detach().cpu()
        del batch
        
        torch.cuda.empty_cache()
        gc.collect()


        return processed_batch