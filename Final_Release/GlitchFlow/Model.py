import torch
import torch.nn as nn
if torch.cuda.is_available():
    device = 'cuda'
    
else:
    device = 'cpu'
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.measure import label
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from . Data import plot_images_gfc

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,dropout_rate=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            #nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x) + self.shortcut(x))

class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, g):
        bs, c, h, w = x.size()
        proj_query = self.query(x).view(bs, -1, h*w).permute(0,2,1)
        proj_key = self.key(g).view(bs, -1, h*w)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(g).view(bs, -1, h*w)
        
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(bs, c, h, w)
        return self.gamma * out + x

class UNet(nn.Module):
    def __init__(self, input_channels=10, output_channels=1, base_channels=64, use_attention=True,encoder_dropout_rate=0.2,decoder_dropout_rate=0.3):
        super().__init__()
        self.use_attention = use_attention
        self._initialize_weights()

        # Encoder
        self.enc1 = nn.Sequential(
            ResidualBlock(input_channels, base_channels,dropout_rate=encoder_dropout_rate),
            ResidualBlock(base_channels, base_channels,dropout_rate=encoder_dropout_rate)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels*2,dropout_rate=encoder_dropout_rate),
            ResidualBlock(base_channels*2, base_channels*2,dropout_rate=encoder_dropout_rate)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            ResidualBlock(base_channels*2, base_channels*4,dropout_rate=encoder_dropout_rate),
            ResidualBlock(base_channels*4, base_channels*4,dropout_rate=encoder_dropout_rate)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels*4, base_channels*8,dropout_rate=decoder_dropout_rate),
            ResidualBlock(base_channels*8, base_channels*8,dropout_rate=decoder_dropout_rate)
        )
        
        # Decoder with or without attention
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1)
        )
        if self.use_attention:
            self.att1 = AttentionGate(base_channels*4)
        self.dec1 = nn.Sequential(
            ResidualBlock(base_channels*8 if self.use_attention else base_channels*4, base_channels*4,dropout_rate=decoder_dropout_rate), # Conditional input channels
            ResidualBlock(base_channels*4, base_channels*4,dropout_rate=encoder_dropout_rate)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1)
        )
        if self.use_attention:
            self.att2 = AttentionGate(base_channels*2)
        self.dec2 = nn.Sequential(
            ResidualBlock(base_channels*4 if self.use_attention else base_channels*2, base_channels*2,dropout_rate=decoder_dropout_rate), # Conditional input channels
            ResidualBlock(base_channels*2, base_channels*2,dropout_rate=decoder_dropout_rate)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1)
        )
        if self.use_attention:
            self.att3 = AttentionGate(base_channels)
        self.dec3 = nn.Sequential(
            ResidualBlock(base_channels*2 if self.use_attention else base_channels, base_channels,dropout_rate=decoder_dropout_rate), # Conditional input channels
            ResidualBlock(base_channels, base_channels,dropout_rate=decoder_dropout_rate)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, 1),
            nn.Softplus() # Output is positive semidefinite
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder
        d1 = self.up1(b)
        if self.use_attention:
            e3 = self.att1(e3, d1)
            d1 = self.dec1(torch.cat([d1, e3], 1))
        else:
            d1 = self.dec1(d1) # No attention, direct input

        d2 = self.up2(d1)
        if self.use_attention:
            e2 = self.att2(e2, d2)
            d2 = self.dec2(torch.cat([d2, e2], 1))
        else:
            d2 = self.dec2(d2) # No attention, direct input
        
        d3 = self.up3(d2)
        if self.use_attention:
            e1 = self.att3(e1, d3)
            d3 = self.dec3(torch.cat([d3, e1], 1))
        else:
            d3 = self.dec3(d3) # No attention, direct input
        
        return self.final(d3)
    
    
#-----------------------------------------TRAINING FUNCTIONS----------------------------------------------------------------------------------------    



def calculate_iou_2d_non0(generated_tensor, target_tensor,norm, threshold=20.0):
    
    #print('norm')
    #print(norm)
    
    
    threshold= threshold/norm
    
    
    
    
    """
    Calculate Intersection over Union (IoU) in the 2D plane at the specified intensity threshold for each element in the batch.

    Parameters:
    - generated_tensor: Tensor containing generated spectrograms (batch_size x 1 x height x width)
    - target_tensor: Tensor containing target spectrograms (batch_size x 1 x height x width)
    - threshold: Intensity threshold for determining the binary masks

    Returns:
    - mean_iou: Mean Intersection over Union (IoU) across all elements in the batch
    - zero_union_count: Count of elements in the batch with a union of 0
    """
    # Convert intensity threshold to tensor
    threshold_tensor = torch.tensor(threshold, device=generated_tensor.device)

    # Create binary masks based on the intensity threshold
    gen_mask = generated_tensor >= threshold_tensor
    tgt_mask = target_tensor >= threshold_tensor

    # Convert masks to float tensors
    gen_mask = gen_mask.float()
    tgt_mask = tgt_mask.float()

    
    # Calculate intersection and union for each element in the batch
    intersection = torch.sum(gen_mask * tgt_mask, dim=(1, 2, 3))
    union = torch.sum(gen_mask, dim=(1, 2, 3)) + torch.sum(tgt_mask, dim=(1, 2, 3)) - intersection

    # Find elements with union 0
    zero_union_mask = union == 0
    zero_union_count = torch.sum(zero_union_mask).item()

    # Exclude elements with union 0 from the IoU calculation
    iou = intersection / union
    iou[zero_union_mask] = 0

    # Take mean over non-zero elements in the batch
    non_zero_count = len(union) - zero_union_count
    mean_iou = torch.sum(iou) / non_zero_count if non_zero_count > 0 else torch.tensor(0.)
    
    
    
    #print(mean_iou)
    #print(mean_iou.shape)
    
    # Count elements with IoU above 0.9
    #above_09_count = torch.sum(iou > 0.9).item()

    return mean_iou#.item()#, zero_union_count, above_09_count
    #return mean_iou.item(), zero_union_count
    
    
   # utils function to generate data using the decoder for inference 
def generate_data(generator, batch):
    """
    Generate data using a generator model.

    Args:
        - generator (nn.Module): Generator model.
        - batch (torch.Tensor): Input batch data.
        

    Returns:
        - torch.Tensor: Generated data.
    """
    #target = batch[:, 0].unsqueeze(1).to(device)
    input = batch[:, 1:].to(device)
    #print('input', input.shape)
    with torch.no_grad():
        generated = generator(input.float()).detach().cpu()
        #print(generated.shape)
        
        
     
    return generated

#----------------accuracy---------------------------------------------------------------------------------------------------------------

class ClusterAboveThreshold(nn.Module):
    def __init__(self, threshold, min_cluster_area):
        super(ClusterAboveThreshold, self).__init__()
        self.threshold = threshold
        self.min_cluster_area = min_cluster_area

    def forward(self, input_tensor):
        # Create a boolean mask based on the threshold
        mask = input_tensor.squeeze(1) >= self.threshold  # Squeeze the channel dimension
        #for i in range(mask.shape[0]):
            #print(torch.count_nonzero(mask[i]))
        
        # Label connected components for the entire batch
        labeled_masks, num_features = label(mask.cpu().numpy(), connectivity=2, return_num=True)

        
        # Reshape labeled_masks to [batch_size, num_features, height, width]
        labeled_masks = torch.tensor(labeled_masks, dtype=torch.long, device=input_tensor.device)
        labeled_masks = labeled_masks.view(input_tensor.size(0), -1, input_tensor.size(-2), input_tensor.size(-1))
        
        # Give unique labels to each cluster acroos each item
        labeled_masks_sorted=self.sort_labels(labeled_masks)
        
        batch_clusters=[]
        for idx, item in enumerate(input_tensor):
            item_clusters=[]
            for cluster in torch.unique(labeled_masks_sorted[idx]):
                if cluster==torch.tensor(0):
                    continue
                cluster_pixels = (labeled_masks_sorted[idx] == cluster)

                # Compute the total area of the cluster
                cluster_area = torch.sum(cluster_pixels)


                # Check if the cluster area is greater than the threshold area
                if cluster_area > self.min_cluster_area:
                    # Flatten the tensor
                    masked_item = item.masked_fill(~cluster_pixels, 0)
                    flattened_tensor = masked_item.flatten()
                    
                    # Compute the maximum value and its index across all dimensions
                    max_value, max_index_flat = torch.max(flattened_tensor, dim=0)

                    # Unravel the flattened index to get the original indices
                    max_index_unraveled = np.unravel_index(max_index_flat.item(), item.shape)
                    item_clusters.append((max_value,max_index_unraveled[1],max_index_unraveled[2]))
            
            batch_clusters.append(item_clusters)   
            

        
        return batch_clusters
    
    def sort_labels(self, labeled_masks):
        # Rename clusters for each item in the batch
        offset = 0
        for i in range(labeled_masks.shape[0]):  # Iterate over batch dimension
            item_labeled_masks = labeled_masks[i]  # Get labeled mask for the current item
            unique_labels = torch.unique(item_labeled_masks)  # Get unique labels in the item's labeled mask
            
            # Exclude background class label (label 0)
            unique_labels = unique_labels[unique_labels != 0]


            # Rename the labels with an offset, starting from 1
            renamed_labels = item_labeled_masks.clone()
            for j, label in enumerate(unique_labels, start=1):
                mask = item_labeled_masks == label
                renamed_labels[mask] = j + offset

            # Update the offset for the next item
            offset += len(unique_labels)

            # Update the labeled mask for the current item
            labeled_masks[i] = renamed_labels

        # labeled_masks now contains the labeled masks with renamed clusters
        return labeled_masks
    
    
def glitch_classifier(list_of_lists):
    list=[1 if sublist else 0 for sublist in list_of_lists ]
    #print(len(list))
    return list
    
def classifier_accuracy(predictions,labels):
    list_check=[(x + y)%2 for x, y in zip(predictions, labels)]
    list_check=np.array(list_check)
    accuracy=1-np.mean(list_check)
    return accuracy
    



def single_analyze_clusters_for_threshold(abs_difference_tensor, generated_tensor, target_tensor, norm_factor,threshold=16, min_cluster_area=1):
        

    cluster_abs_diff_accuracies = []
    clusters_generated_accuracies = []

    

    # pipeline
    cluster_nn = ClusterAboveThreshold(threshold, min_cluster_area).to('cpu')  # Assuming to('cpu') is needed

    # get clusters
    clusters_abs_diff = cluster_nn(abs_difference_tensor)
    clusters_generated = cluster_nn(generated_tensor * norm_factor)
    clusters_target = cluster_nn(target_tensor * norm_factor)

    
    #set labels
    target_labels = glitch_classifier(clusters_target)  # Use target clusters as labels
    #print("TEST")
    #print("TEST")
    #print("TEST")
    #print(target_labels)
    #print(len(target_labels))
            
    diff_labels= [0 for k in range(len(target_labels))]
        
    # Calculate classifier accuracy for abs_difference_tensor
    abs_diff_predictions = glitch_classifier(clusters_abs_diff)
    abs_diff_accuracy = classifier_accuracy(abs_diff_predictions, diff_labels)
    cluster_abs_diff_accuracies.append(abs_diff_accuracy)

    # Calculate classifier accuracy for generated_tensor
    generated_predictions = glitch_classifier(clusters_generated)
    generated_accuracy = classifier_accuracy(generated_predictions, target_labels)
    clusters_generated_accuracies.append(generated_accuracy)

    return cluster_abs_diff_accuracies, clusters_generated_accuracies
    
    
#-----------------------------------------------------------------------------------------------------------------------------------------------    
    


def train_decoder(num_epochs, 
                  generator, 
                  criterion1, 
                  optimizer, 
                  dataloader, 
                  test_set, 
                  background_set,
                  channel_means, 
                  checkpoint_path,
                  accuracy=single_analyze_clusters_for_threshold,
                  snr2_threshold=16,
                  save_best=True, 
                  scheduler=None, 
                  max_grad_clip=5.0, 
                  logger=None,
                  nacc=1,
                  acc_batch=10):
    """
    Trains the generator model using a combination of two loss functions with dynamic weighting.

    Args:
        num_epochs: (int) Number of epochs for training.
        generator: (NN.Module) NN model to train.
        criterion1: (CustomLoss) Primary loss function (e.g., L1).
        criterion2: (CustomLoss) Secondary loss function (e.g., L2).
        optimizer: (torch.optim) Optimizer for training.
        dataloader: (DataLoader) Training data loader.
        val_loader: (DataLoader) Validation data loader.
        accuracy: (function) Metric to measure performance of the model.
        
        checkpoint_path: (str) Path to save checkpoints.
        save_best: (bool) Whether to save the best performing model.
        scheduler: (torch.optim.lr_scheduler) Learning rate scheduler.
        switch_threshold: (float) Minimum change in loss to consider as progress.
        patience: (int) Number of epochs to wait before switching loss priority.
    
    Returns:
        loss_plot, val_loss_plot: Training and validation loss history.
    """
    
    # Initialize tracking metrics
    loss_plot = []
    val_loss_plot = []
    best_val_loss = float('inf')
    
    
    
    
    # Initialize Data for accuracy calculation
    
    val_loader = DataLoader(
           test_set,
           batch_size=acc_batch,
           shuffle=False,
        )
    
    back_loader = DataLoader(
           background_set,
           batch_size=acc_batch,
           shuffle=False,
        )
    
    
    norm=torch.tensor(channel_means[0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    target_tensor=torch.cat((test_set[:,0,:,:].unsqueeze(1),background_set[:,0,:,:].unsqueeze(1)))
        
    
    
    
    
    
    logger.create_logger_context()
    
    
    
    if scheduler is not None:
        print(f'{scheduler=}')
    
    for epoch in tqdm(range(1, num_epochs + 1)):
        generator.train()  # Set model to training mode
        epoch_loss = []

        for i, batch in enumerate(dataloader):
            torch.cuda.empty_cache()
            target = batch[:, 0].unsqueeze(1).to(device).float()
            input = batch[:, 1:].to(device)

            optimizer.zero_grad()  # Zero the gradients
            generated = generator(input.float())  # Forward pass

            # Compute both loss components
            total_loss = criterion1(generated, target)


            
            total_loss.backward()  # Backpropagation
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_clip)
            
            optimizer.step()  # Update model parameters

            epoch_loss.append(total_loss.detach().cpu().numpy())
            

        # Validation phase
        generator.eval()  # Set model to evaluation mode
        val_loss = []
        with torch.no_grad():
            for batch in val_loader:
                torch.cuda.empty_cache()
                target = batch[:, 0].unsqueeze(1).to(device).float()
                input = batch[:, 1:].to(device)
                generated = generator(input.float())

                # Compute validation losses
                total_val_loss = criterion1(generated, target)     
                 
                val_loss.append(total_val_loss.detach().cpu().numpy())

        # Record training and validation loss
        #log tensorboard
        loss_plot.append(np.mean(epoch_loss))
        logger.log(np.mean(epoch_loss),"Training Loss",step=epoch)
        val_loss_plot.append(np.mean(val_loss))
        logger.log(np.mean(val_loss),"Validation Loss",step=epoch)
        
        # Adjust learning rate using scheduler
        if scheduler is not None:
            scheduler.step(val_loss_plot[-1])

        # Print progress
        print(f'Epoch {epoch}: training loss {loss_plot[-1]:.4e}, val loss {val_loss_plot[-1]:.4e}')

        # Improvement check (check if the loss has stagnated)
        if epoch > 1:
            improvement = (val_loss_plot[-2] - val_loss_plot[-1]) / val_loss_plot[-2]
            print(f'Improvement: {improvement*100:.4f}%')


        # Save checkpoint if validation loss improves
        if save_best and val_loss_plot[-1] < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_plot[-1],
                'val_loss': val_loss_plot[-1],
            }
            best_val_loss = val_loss_plot[-1]
            torch.save(checkpoint, checkpoint_path.format('best'))
            
        # Evaluate accuracy every epoch
        if epoch % nacc == 0 and (not(os.path.isfile("./conf/accuracy.lock"))):
            
            print('Logging accuracies.')
            
            
            
            generated_tensor_pre = torch.tensor([]).to('cpu')  # Initialize an empty tensor
            for batch in val_loader:
                generated_post = generate_data(generator, batch.detach().cpu()).to('cpu')
                generated_tensor_pre = torch.cat((generated_tensor_pre, generated_post), dim=0)
            
            background_tensor = torch.tensor([]).to('cpu')  # Initialize an empty tensor
            for batch in back_loader:
                background_post = generate_data(generator, batch.detach().cpu()).to('cpu')
                background_tensor = torch.cat((background_tensor, background_post), dim=0)
                
            generated_tensor=torch.cat((generated_tensor_pre,background_tensor), dim=0)    
            
            
        
            abs_difference_tensor=torch.abs((generated_tensor-target_tensor)*norm)
            
            cluster_abs_diff_accuracies, clusters_generated_accuracies = accuracy(abs_difference_tensor,                                                                                                                                 generated_tensor,
                                                                                    target_tensor, 
                                                                                    norm,
                                                                                    threshold=snr2_threshold)
            
            
            

            
            
            logger.log(cluster_abs_diff_accuracies[0],f"Denoising Accuracy SNR^2 {snr2_threshold}",step=epoch)
            logger.log(clusters_generated_accuracies[0],f"Veto Accuracy SNR^2 {snr2_threshold}",step=epoch)
            gfc=plot_images_gfc(generated_tensor,test_set , channel_means,num_aux_channels=(test_set.shape[1]-1))
            #Log to tensorboard
            logger.log(gfc,"Sample",kind="figure",step=epoch)
            #print(f'Epoch {epoch}: Validation accuracy: {avg_accuracy:.4f}')

    return loss_plot, val_loss_plot


#---------------------------------------------LOSS COMPUTATION-------------------------------------------------

class MeanAbsDiff(nn.Module):
    def __init__(self):
        super(MeanAbsDiff, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.abs(y_pred - y_true)
        return loss.mean()

class StdAbsDiff(nn.Module):
    def __init__(self):
        super(StdAbsDiff, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.abs(y_pred - y_true)
        return loss.std()
    
    
def calculate_single_loss(generator, criterion, val_loader):
    generator.eval()  # Set the model to evaluation mode

    val_total_loss = []  # To store total losses

    with torch.no_grad():
        for batch in val_loader:
            torch.cuda.empty_cache()
            target = batch[:, 0].unsqueeze(1).to(device)
            input_ = batch[:, 1:].to(device)
            generated = generator(input_)

            # Get the individual loss components from the criterion
            total_loss = criterion(generated, target)

            val_total_loss.append(total_loss.item())

    # Return mean of the losses and the full lists

    mean_total_loss = np.mean(val_total_loss)

    return mean_total_loss, val_total_loss



class CustomLoss(nn.Module):
    def __init__(self, alpha=0.8, data_range=21.0):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        self.alpha = alpha

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * ssim_loss

