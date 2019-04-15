import os
import torch

def save_model(model, optimizer, epoch, checkpoint_prefix):
    checkpoint_name = '{}.tar'.format(checkpoint_prefix)
    checkpoint_path = os.path.join('checkpoints', checkpoint_name)
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, checkpoint_path)