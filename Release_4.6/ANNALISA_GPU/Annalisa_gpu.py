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
from .qtransform_gpu import *
from ml4gw.transforms import SpectralDensity,Whiten
from  .Peak_finder_torch import * 

if torch.cuda.is_available():
    device = 'cuda'
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    device = 'cpu'
print(f'Running on device: {device=}')


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
                 logf=True, qtile_mode=False,whiten=False,psd=None):
        super(QT_dataset, self).__init__()
        # Set device
        self.device = device
        
        # whitening parameters
        self.psd=psd
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
            spectrogram_shape=(self.num_f_bins, self.num_t_bins),
            logf=self.logf,
            qtiles_mode=self.qtile_mode
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
                
        qt_batch = self.qtransform(batch.to(self.device))


        return qt_batch.detach().cpu()
