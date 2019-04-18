import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_l2dist(prediction, targets):
    """
    Calculate reconstruction error by calculating L2-norm for (x, y, t)
    """
    assert prediction.size() == targets.size()
    # (batch_size, channel, height, width)
    error = torch.pow(targets - prediction, 2)
    error = torch.sum(error, dim=(1, 2, 3))
    error = torch.sqrt(error)
    # (batch_size,)
    return error

def calculate_psnr(prediction, targets):
    """
    Calculate reconstruction error by calculating L2-norm for (x, y, t)
    """
    assert prediction.size() == targets.size()
    # (batch_size, channel, height, width)
    error = torch.pow(targets - prediction, 2)
    error = error.mean((1, 2, 3)) # MSE
    error = 10 * torch.log10(1 / error)
    # (batch_size, )
    return error