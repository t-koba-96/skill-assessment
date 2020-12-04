import os
import shutil
import torch


def make_dirs(args, opts, train_txt, val_txt, savedir):
    train_list = os.path.join(args.data_path, opts.dataset, 'splits', opts.task, train_txt)
    valid_list = os.path.join(args.data_path, opts.dataset, 'splits', opts.task, val_txt)
    feature_path = os.path.join(args.data_path, opts.dataset, 'features', opts.task)
    writedir = os.path.join(args.writer_path, savedir)
    ckptdir = os.path.join(args.ckpt_path, savedir)
    dir_list = [writedir, ckptdir]
    for dir in dir_list:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
    return train_list, valid_list, feature_path, writedir, ckptdir


def accuracy(score_pos, score_neg):
    """Computes the % of correctly ordered pairs"""
    pred1 = score_pos
    pred2 = score_neg
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
