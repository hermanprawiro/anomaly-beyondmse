import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import time
import torch
import torch.nn as nn
import torchvision

from dataset.ucsdped import UCSDPedDataset, UCSDPedFlowDataset
from utils.metrics import AverageMeter
from sklearn.metrics import roc_auc_score, roc_curve

from models import networks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GDLLoss(nn.Module):
    def __init__(self, n_channels=1, alpha=1):
        super().__init__()

        pos = torch.tensor(np.identity(n_channels), dtype=torch.float, device=device)
        neg = -pos
        self.alpha = alpha
        self.filter_x = torch.stack((neg, pos), dim=-1).unsqueeze(-2)
        self.filter_y = torch.stack((neg, pos), dim=-1).unsqueeze(-1)

    def forward(self, gen_frames, gt_frames):
        gen_dx = torch.abs(nn.functional.conv2d(gen_frames, self.filter_x, stride=1, padding=0))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames, self.filter_y, stride=1, padding=0))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames, self.filter_x, stride=1, padding=0))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames, self.filter_y, stride=1, padding=0))

        grad_x = torch.abs(gt_dx - gen_dx) ** self.alpha
        grad_y = torch.abs(gt_dy - gen_dy) ** self.alpha

        return torch.mean(torch.sum(grad_x, dim=(1, 2, 3)) + torch.sum(grad_y, dim=(1, 2, 3)))

def main():
    root_dir_ped1 = "D:\\Datasets\\UCSD Anomaly\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1"
    root_dir_ped2 = "D:\\Datasets\\UCSD Anomaly\\UCSD_Anomaly_Dataset.v1p2\\UCSDped2"
    root_flow_ped1 = "E:\\Datasets\\UCSD_Flow_TVL1\\UCSDped1"
    root_flow_ped2 = "E:\\Datasets\\UCSD_Flow_TVL1\\UCSDped2"

    learning_rate = 5e-4
    train_batch_size = 8
    test_batch_size = 32
    start_epoch = 0
    max_epoch = 250

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])

    # train_set = UCSDPedFlowDataset(root_dir=root_dir_ped1, flow_root=root_flow_ped1, train=True, frame_length=2, frame_stride=1, transforms=transforms, dimension=2)
    # test_set = UCSDPedFlowDataset(root_dir=root_dir_ped1, flow_root=root_flow_ped1, train=False, frame_length=2, frame_stride=1, transforms=transforms, dimension=2)
    train_set = UCSDPedDataset(root_dir=root_dir_ped2, train=True, frame_length=5, frame_stride=1, transforms=transforms, dimension=2)
    test_set = UCSDPedDataset(root_dir=root_dir_ped2, train=False, frame_length=5, frame_stride=1, transforms=transforms, dimension=2)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=4)

    netG = networks.define_G(4, 1, 32, 'unet_128').to(device)
    netD = networks.define_D(1, 64, 'basic').to(device)

    rec_criterion = nn.MSELoss().to(device)
    gan_criterion = networks.GANLoss('lsgan').to(device)
    gdl_criterion = GDLLoss().to(device)

    optG = torch.optim.Adam(netG.parameters(), lr=1e-3)
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4)

    checkpoint_prefix = 'unet128_32G_patchganD_LSGANLoss_GDLLoss-L2_1scale'
    checkpoint_path = os.path.join('checkpoints', '{}.tar'.format(checkpoint_prefix))
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        netG.load_state_dict(checkpoint['state_dict']['generator'])
        netD.load_state_dict(checkpoint['state_dict']['discriminator'])
        optG.load_state_dict(checkpoint['optimizer']['generator'])
        optD.load_state_dict(checkpoint['optimizer']['discriminator'])
        print('Checkpoint loaded, last epoch = {}'.format(checkpoint['epoch'] + 1))
        start_epoch = checkpoint['epoch'] + 1

    result_path = os.path.join('results', checkpoint_prefix)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for epoch in range(start_epoch, max_epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        R_losses = AverageMeter()
        GDL_losses = AverageMeter()
        G_losses = AverageMeter()
        D_losses = AverageMeter()

        netD.train()
        netG.train()

        total_iter = len(train_loader)
        end = time.time()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            inputs, future = torch.split(inputs, 4, dim=1)

            # measure data loading time
            data_time.update(time.time() - end)

            batch_size = inputs.size(0)
            # Reconstruction
            optD.zero_grad()
            optG.zero_grad()
            
            # outputs = netG(inputs)
            # lossR = rec_criterion(outputs, future)
            
            # lossR.backward()
            # netG.step()
            fake_recon = netG(inputs)

            # Discriminator
            # Fake
            optD.zero_grad()
            fake_pred = netD(fake_recon.detach())
            DGz1 = fake_pred.mean().item()
            lossD_fake = gan_criterion(fake_pred, False)

            # Real
            real_pred = netD(future)
            Dx = real_pred.mean().item()
            lossD_real = gan_criterion(real_pred, True)
            lossD = (lossD_real + lossD_fake) * 0.5
            lossD.backward()
            optD.step()

            # Generator
            optG.zero_grad()
            fake_pred = netD(fake_recon)
            DGz2 = fake_pred.mean().item()
            lossG_gan = gan_criterion(fake_pred, True)
            lossG_recon = rec_criterion(fake_recon, future)
            lossG_gdl = gdl_criterion(fake_recon, future)
            lossG = lossG_gan + lossG_recon + lossG_gdl
            lossG.backward()
            optG.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # measure accuracy and record loss
            R_losses.update(lossG_recon.item(), batch_size)
            GDL_losses.update(lossG_gdl.item(), batch_size)
            D_losses.update(lossD.item(), batch_size)
            G_losses.update(lossG.item(), batch_size)

            # global_step = (epoch * total_iter) + i + 1
            # writer.add_scalar('train/loss', losses.val, global_step)

            if i % 10 == 0:
                print('Epoch {0} [{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss_R {lossR.val:.4f} ({lossR.avg:.4f})\t'
                    'Loss_GDL {lossGDL.val:.4f} ({lossGDL.avg:.4f})\t'
                    'Loss_D {lossD.val:.4f} ({lossD.avg:.4f})\t'
                    'Loss_G {lossG.val:.4f} ({lossG.avg:.4f})\t'
                    'D(x) {dx:.4f}\tD(G(z)) {dgz1:.4f} / {dgz2:.4f}'
                    .format(
                    epoch + 1, i + 1, total_iter, 
                    batch_time=batch_time, data_time=data_time, lossR=R_losses, lossGDL=GDL_losses,
                    lossD=D_losses, lossG=G_losses,
                    dx=Dx, dgz1=DGz1, dgz2=DGz2
                    )
                )

        save_model((netG, netD), (optG, optD), epoch, checkpoint_prefix)
        test(test_loader, netG)
        
        with torch.no_grad():
            testiter = iter(test_loader)
            inputs, _ = next(testiter)
            inputs, _ = next(testiter)
            inputs, _ = next(testiter)
            inputs, _ = next(testiter)
            inputs, future = torch.split(inputs, 4, dim=1)

            fake = netG(inputs.to(device)).detach().cpu()
            torchvision.utils.save_image(torch.cat((future, fake), 0), os.path.join(result_path, '{:03d}.png'.format(epoch)), nrow=8, padding=2, normalize=True)

def save_model(models, optimizers, epoch, checkpoint_prefix):
    checkpoint_name = '{}.tar'.format(checkpoint_prefix)
    checkpoint_path = os.path.join('checkpoints', checkpoint_name)
    netG, netD = models
    optG, optD = optimizers
    torch.save({
        'state_dict': {
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
        },
        'optimizer': {
            'generator': optG.state_dict(),
            'discriminator': optD.state_dict(),
        },
        'epoch': epoch,
    }, checkpoint_path)

def train(loader, model, optimizer, criterion, writer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    R_losses = AverageMeter()

    encoder, decoder = model
    opt_enc, opt_dec = optimizer
    rec_criterion, adv_criterion = criterion

    encoder.train()
    decoder.train()

    total_iter = len(loader)
    end = time.time()
    for i, (inputs, flows, targets) in enumerate(loader):
        inputs = inputs.to(device)
        flows = flows.to(device)
        inputs, future = torch.split(inputs, 1, dim=1)
        inputs = torch.cat((inputs, flows.permute(0, 3, 1, 2)), 1)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs.size(0)
        # Reconstruction
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        
        outputs = decoder(encoder(inputs))
        lossR = rec_criterion(outputs, future)
        
        lossR.backward()
        opt_dec.step()
        opt_enc.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        R_losses.update(lossR.item(), batch_size)

        # global_step = (epoch * total_iter) + i + 1
        # writer.add_scalar('train/loss', losses.val, global_step)

        if i % 10 == 0:
            print('Epoch {0} [{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss_R {lossR.val:.4f} ({lossR.avg:.4f})\t'
                .format(
                epoch + 1, i + 1, total_iter, 
                batch_time=batch_time, data_time=data_time, lossR=R_losses, 
                )
            )

def calculate_error(prediction, targets):
    """
    Calculate reconstruction error by calculating L2-norm for (x, y, t)
    """
    assert prediction.size() == targets.size()
    # (batch_size, window_size, height, width)
    error = torch.pow(targets - prediction, 2)
    error = torch.sum(error, dim=(-1, -2, -3))
    error = torch.sqrt(error)
    return error.cpu().numpy()

def expand_results(raw_results):
    expanded_result = []
    previous_id = 0
    previous_errors = []
    for video_id, clip_error in raw_results:
        if video_id != previous_id:
            # previous_errors = imresize(np.expand_dims(previous_errors, axis=1), (len(previous_errors) * 5, 1), interp='bicubic', mode='F').flatten()
            expanded_result.append(previous_errors)
            previous_errors = []
        previous_errors.append(clip_error)
        previous_id = video_id
    expanded_result.append(previous_errors)
    return expanded_result

def calculate_auc(results):
    gt_all = pickle.load(open(os.path.join(os.getcwd(), 'dataset', 'ucsd', 'ped2_gt.pkl'), 'rb'))
    unused_frames = 4

    y_true = []
    y_pred = []
    for result, gts in zip(results, gt_all):
        gt = np.ones_like(result)
        result = result - np.min(result)
        result = 1 - result / np.max(result)
        y_pred.append(result)
        # ax = plt.gca()
        for gt_id in gts:
            gt[gt_id[0] - 1 - unused_frames: gt_id[1] - unused_frames] = 0
            # ax.add_patch(matplotlib.patches.Rectangle((gt_id[0] - 1 - unused_frames, 0), gt_id[1] - gt_id[0], 1, facecolor='red', alpha=0.4))
        y_true.append(gt)
        # plt.plot(np.arange(result.shape[0]), result)
        # plt.ylim(0, 1)
        # plt.show()
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    # print(y_true.shape, y_pred.shape)
    return roc_auc_score(y_true, y_pred)

@torch.no_grad()
def test(loader, models):
    models.eval()

    total_iter = len(loader)
    errors_all = []
    targets_all = []

    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        inputs, future = torch.split(inputs, 4, dim=1)

        outputs = models(inputs)
        error = calculate_error(outputs, future)
        
        errors_all.append(error)
        targets_all.append(targets.cpu().numpy())

    errors_all = np.concatenate(errors_all)
    targets_all = np.concatenate(targets_all)

    results = list(zip(targets_all, errors_all))
    results = expand_results(results)
    auc = calculate_auc(results)
    print('> AUC = {:.5f}'.format(auc))


if __name__ == "__main__":
    main()