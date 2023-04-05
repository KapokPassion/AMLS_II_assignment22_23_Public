import os
import copy
import time
import h5py

import numpy as np
from torch import Tensor
import torch
from torch import nn
import torch.optim as optim

# gpu acc
import torch.backends.cudnn as cudnn

from torch.utils.data.dataloader import DataLoader

from models import FSRCNN, OFSRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, ssim

def train(scale, patch_size, model_name, num_epochs, continue_epoch=0, batch_size=256):

    train_file = r'./Datasets/train_X{}.h5'.format(scale)
    eval_file = r'./Datasets/valid_X{}.h5'.format(scale)
    outputs_dir = r'./outputs'
    log_dir = r'./log'
    lr_1 = 1e-3
    lr_2 = 1e-3
    lr_3 = 1e-3
    lr_4 = 1e-3
    lr_5 = 1e-4

    batch_size = 256
    num_workers = 0
    num_epochs = num_epochs
    seed = 1
    model_name = model_name
    continue_epoch = continue_epoch # will load this epoch weight file to continue

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    # benckmark mode to acc
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)


    model = FSRCNN(scale=scale).to(device) if model_name == 'FSRCNN' else OFSRCNN(scale=scale).to(device)

    # plt
    lossLog = []
    psnrLog = []
    ssimLog = []

    if continue_epoch != 0:
        model.load_state_dict(torch.load(os.path.join(outputs_dir, '{}_X{}_epoch_{}.pth'.format(model_name, scale, continue_epoch))))
        lossLog = np.loadtxt(os.path.join(log_dir, '{}_X{}_lossLog.txt'.format(model_name, scale)))
        lossLog = lossLog.tolist()
        psnrLog = np.loadtxt(os.path.join(log_dir, '{}_X{}_psnrLog.txt'.format(model_name, scale)))
        psnrLog = psnrLog.tolist()
        ssimLog = np.loadtxt(os.path.join(log_dir, '{}_X{}_ssimLog.txt'.format(model_name, scale)))
        ssimLog = ssimLog.tolist()


    # loss MSE
    criterion = nn.MSELoss()

    # opt
    optimizer = optim.Adam(
        [{'params': model.features.parameters(), 'lr': lr_1},
         {'params': model.shrinking.parameters(), 'lr': lr_2},
         {'params': model.mapping.parameters(), 'lr': lr_3},
         {'params': model.expanding.parameters(), 'lr': lr_4},
         {'params': model.deconv.parameters(), 'lr': lr_5},
        ]) if model_name == 'FSRCNN' else optim.Adam(
        [{'params': model.features.parameters(), 'lr': lr_1},
         {'params': model.shrinking.parameters(), 'lr': lr_2},
         {'params': model.mapping.parameters(), 'lr': lr_3},
         {'params': model.expanding.parameters(), 'lr': lr_4},
         {'params': model.upsample.parameters(), 'lr': lr_5},
        ])

    train_dataset = TrainDataset(h5_file=train_file, patch_size=patch_size, scale=scale)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        # dataset has already shuffled
        shuffle=False,
        num_workers=num_workers,
    #     pin_memory=True,
        drop_last=True)

    eval_dataset = EvalDataset(h5_file=eval_file, patch_size=patch_size, scale=scale)
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True)

    # weights copy
    best_psnr = 0.0
    best_epoch = 0
    if continue_epoch != 0:
        model.load_state_dict(torch.load(os.path.join(outputs_dir, '{}_X{}_best.pth'.format(model_name, scale))))
    best_weights = copy.deepcopy(model.state_dict())
    best_psnr = max(psnrLog)
    best_epoch = psnrLog.index(best_psnr) + 1

    # Train
    for epoch in range(continue_epoch + 1, continue_epoch + num_epochs + 1):
        since = time.time()
        print('epoch: '+ str(epoch) + ' at ' + time.strftime("%H:%M:%S"))

        model.train()

        epoch_losses = AverageMeter()

        process = 0
        for data in train_dataloader:
            process += 1
            print('\r', '***training process of epoch {} : {:.2f}%***'.format(epoch, process / len(train_dataset) * batch_size * 100), end='')

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        lossLog.append(np.array(epoch_losses.avg))
        print('')
        print('train loss: ' + str(epoch_losses.avg))
        np.savetxt(os.path.join(log_dir, '{}_X{}_lossLog.txt'.format(model_name, scale)), lossLog)

        torch.save(model.state_dict(), os.path.join(outputs_dir, '{}_X{}_epoch_{}.pth'.format(model_name, scale, epoch)))

        # PSNR SSIM
        model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        process = 0
        for data in eval_dataloader:
            process += 1
            print('\r', '***eval process of epoch {} : {:.2f}%***'.format(epoch, process / len(eval_dataset) * batch_size * 100), end='')
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
    #             preds = model(inputs)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_ssim.update(ssim(preds, labels), len(inputs))

        print('')
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        print('eval ssim: {:.2f}'.format(epoch_ssim.avg))

        psnrLog.append(Tensor.cpu(epoch_psnr.avg))
        ssimLog.append(Tensor.cpu(epoch_ssim.avg))
        np.savetxt(os.path.join(log_dir, '{}_X{}_psnrLog.txt'.format(model_name, scale)), psnrLog)
        np.savetxt(os.path.join(log_dir, '{}_X{}_ssimLog.txt'.format(model_name, scale)), ssimLog)

        # update weight
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))

        torch.save(best_weights, os.path.join(outputs_dir, '{}_X{}_best.pth'.format(model_name, scale)))

        time_elapsed = time.time() - since
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('')

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))

    torch.save(best_weights, os.path.join(outputs_dir, '{}_X{}_best.pth'.format(model_name, scale)))