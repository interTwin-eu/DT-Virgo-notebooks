import torch
from matplotlib import pyplot as plt


device = 'cuda'
NUM_EPOCHS = 100
OUTPUT_CHANNELS = 3
LAMBDA = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
checkpoint_dir = './save_states/'

def save_checkpoint(models, optimizers, curr_epoch, filename):
    print("=> Saving Checkpoint")
    checkpoint = {
        "gen_state": models[0].state_dict(),
        "gen_opt": optimizers[0].state_dict(),
        "disc_state": models[1].state_dict(),
        "disc_opt": optimizers[1].state_dict(),
        "epoch": curr_epoch
    }
    directory = checkpoint_dir
    extension = '.pth'
    torch.save(checkpoint, directory + filename + extension)


def load_checkpoint(checkpoint_file, models, optimizers, lr=lr):
    directory = checkpoint_dir
    checkpoint = torch.load(directory + checkpoint_file, map_location=device)
    models[0].load_state_dict(checkpoint["gen_state"])
    models[1].load_state_dict(checkpoint["disc_state"])
    optimizers[0].load_state_dict(checkpoint["gen_opt"])
    optimizers[1].load_state_dict(checkpoint["disc_opt"])
    
    epoch = checkpoint["epoch"]
    
    print(f"=> Loading  Checkpoint from epoch {epoch}")
    
    for param_group in optimizers[0].param_groups:
        param_group["lr"] = lr
    for param_group in optimizers[1].param_groups:
        param_group["lr"] = lr
    
    return epoch


def show_images(test_input, tar, prediction):
    fig ,axs = plt.subplots(1, 3, figsize=(7,2))

    axs[0].imshow(test_input[0].detach().permute(1, 2, 0).cpu() * 0.5 + 0.5)
    axs[0].set_title("Observed Image", fontsize=8)
    axs[1].imshow(tar[0].detach().permute(1, 2, 0).cpu() * 0.5 + 0.5)
    axs[1].set_title("Ground Truth", fontsize=8)
    axs[2].imshow(prediction[0].detach().permute(1, 2, 0).cpu() * 0.5 + 0.5)
    axs[2].set_title("Generated Output", fontsize=8)
    
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')