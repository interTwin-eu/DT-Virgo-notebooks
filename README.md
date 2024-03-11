# InterTwin WP 4.4


## Glitch Flow
Non-Gaussian transient noise artifacts, commonly referred to as glitches, are one of the most challenging limitations in the study of gravitational-wave interferometer data due to their similarities with astrophysical sources signals in the time and frequency domains. Therefore, exploring novel methods to recover physical information from data corrupted by glitches is essential. 
We address the issue with Glitch FLow, a Digital Twin (DT) whose purpose is modeling and generating glitches using deep generative algorithms. Namely, we employ Decoder and Encoder Decoder Convolutional architectures for data-to-data translation. This strategy involves mapping glitches from carefully chosen auxiliary channels (uncorrelated with the physical signals) to the 'strain' (main) channel, allowing us to subtract the generated noise from the physically interesting data. 


## Description
This guide illustrates the all the necessary seps to run and customise interTwin_wp_4.4_synthetic_data.ipynb notebook.


## Installation
To run the notebook, it is necessary to install jupyter notebook on your machine, as well as all python packages included in the Import section of th notebook. To train the Neural Networks (NNs) the use of GPU is strongly recomended.

## Usage
The notebook is divided in 6 macro sections:

- ### Import :
In this section all relevant python packages are imported and the device is set to GPU if available.
- ### Create Synthetic Dataset:
In this section, we create a synthetic dataset to train generative models with.

The section is split in two parts:

   1. **TimeSeries Dataset** Generation of a gwpy TimeSeries dataset making use of random noise and sinusoidal functions
   2. **Q-plot Dataset** Conversion of TimeSeries dataset into 2D Image dataset making use of q_transform


The dataset used to train the NN with is created as a 2D images. Note that you do not need to run the two sections each time, but can rather save the dataset after creating it once and loading it at the beginning of Process Data that to save time 

- ### Preprocess Data:
In this section we prepare the dataset for NN training and inference.

The section is divided in two parts:
1. **Split Data**, where we convert the dataset to torch, and then divide it into train and test set (making also a smaller version of the two)
2. **Normalise Data & Dataloader**, where we bring the dataset to the range [0,1] (for NN convergence reasons) and create dataloader objects

- ### NN Models:
In this section we define different NN architectures models, and initialise one of them as the generator to use in training and inference.

This section is split in three parts:
1. **Weight Initialization**, where we define the function to initialise the weights of the NN models according to certain parameters and distributions passed as input
2. **NN Models**, where we define different NN models exploting different architecutres
3. **Generator**, where we initialise one of the above models as the generator to use in training and inference

- ### Training:
  In this section, we train the previously defined and initialised NN model.

This section is divided into three parts:
1. **Functions**, which contains utils functions to calculate several loss functions for the networks, a metric for accuracy (not used in the current version of the notebook) a function to make inference and a function to train the model and save the weights
2. **Pre-training generation**, where we make inference on test data using untrained network
3. **Actual training**, where we train the NN, save the weigths and plot losses curves

- ### Inference:
In this section we make inference on test dataset using trained NN, and we plot the generated qplots.

This section is devided in two parts:
1. **Load Model**, where we load the model from checkpoint
2. **Actual Inferece**, where we generate data for main channel from the test dataset. We also plot the generated data and compare it to the target


## Support
For support contact lorenzo.asprea@to.infn.it and francesco.sarandrea@to.infn.it

## Roadmap


## Contributing


## Authors and acknowledgment
Sara Vallero, Federica Legger, Francesco Sarandrea, Lorenzo Asprea

## License


## Project status
Ongoing
