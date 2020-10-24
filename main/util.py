import os
import shutil
import torch
import glob

from torch.utils.tensorboard import SummaryWriter



def make_dirs(args, opts, train_txt, val_txt, savedir):
    train_list = os.path.join(args.data_path, opts.dataset, 'splits', opts.task, train_txt)
    val_list = os.path.join(args.data_path, opts.dataset, 'splits', opts.task, val_txt)
    root_path = os.path.join(args.data_path, opts.dataset, 'features', opts.task)
    writedir = os.path.join(args.writer_path, savedir)
    ckptdir = os.path.join(args.ckpt_path, savedir)
    dir_list = [writedir, ckptdir]
    for dir in dir_list:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
    writer = SummaryWriter(writedir)
    return train_list, val_list, root_path, writer, ckptdir


def save_checkpoint(state, file_path, epoch, score, ckpt=True, is_best= True):
    if ckpt:
        savefile = os.path.join(file_path, "epoch_{:04d}_acc_{:.4f}.ckpt".format(epoch, score))
        torch.save(state, savefile)
        print(('Saving checkpoint file ... epoch_{:04d}_acc_{:.4f}.ckpt'.format(epoch, score)))
    if is_best:
        if glob.glob(os.path.join(file_path, 'best_score*')):
            os.remove(glob.glob(os.path.join(file_path, 'best_score*'))[0])
        best_name = os.path.join(file_path, "best_score_epoch_{:04d}_acc_{:.4f}.ckpt".format(epoch, score))
        torch.save(state, best_name)
        print(('Saving checkpoint file ... best_score_epoch_{:04d}_acc_{:.4f}.ckpt'.format(epoch, score)))


def accuracy(output1, output2):
    """Computes the % of correctly ordered pairs"""
    pred1 = output1
    pred2 = output2
    correct = torch.gt(pred1, pred2)
    return float(correct.sum())/correct.size(0), int(correct.sum())


def data_augmentation(input_var1, input_var2, device):
    noise = torch.autograd.Variable(torch.normal(torch.zeros(input_var1.size()[1],
                                                             input_var1.size()[2]),
                                                 0.01)).to(device)
    input_var1 = torch.add(input_var1, noise)
    input_var2 = torch.add(input_var2, noise)
    return input_var1, input_var2


class AverageMeter(object):
    """Compute and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset_val(self):
        self.val = 0

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
        

def sec2str(sec):
    if sec < 60:
        return "{:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "{:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "{:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "{:02d} days, {:02d}h{:02d}m{:02d}s".format(dy, hr, min, sec)
