import argparse
import yaml
import torch
import time
import os

from addict import Dict

from main.util import make_dirs
from main.dataset import SkillDataSet
from main.model import RAAN
from main.evalrun import Eval_Runner

'''
default == using 'cuda:0'
'''

def get_arguments():

    parser = argparse.ArgumentParser(description='Evaluation network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('task', type=str, help='choose task(e.g. "drawing" for EPIC-Skills, "origami" for BEST)')
    parser.add_argument('lap', type=str, help='train lap name(count)')
    parser.add_argument('--epoch', type=str, default='best', help='Model weight [best | epoch_num ] ')
    parser.add_argument('--dataset', type=str, default='BEST', help='choose dataset [ EPIC-Skills | BEST ] ')
    parser.add_argument('--split', type=str, default= '1', help='for EPIC-Skills')
    parser.add_argument('--cuda', type=str, default= [0], help='choose cuda num')
    return parser.parse_args()



def main():

    # ====== Args ======
    # start time
    start_time = time.time()

    # parser args
    opts = get_arguments()
    # yaml args
    args = Dict(yaml.safe_load(open(os.path.join('args',opts.arg+'.yaml'))))
    args.start_time = start_time
    input_size = {"1d": 1024, "2d": 512}
    args.input_size = input_size[args.input_feature]
    # show args 
    print(('\n''[Options]\n''{0}\n''\n'
           '[Arguements]\n''{1}\n''\n'.format(opts, args)))

    # device setting
    opts.cuda = list(map(str,opts.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(opts.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ====== Paths ======
    # BEST dataset
    if opts.dataset == "BEST":
        train_txt = "train.txt"
        val_txt = "test.txt"
        savedir = os.path.join(opts.dataset, opts.task, opts.arg, "lap_"+opts.lap)

    # Epic-skills dataset
    elif opts.dataset == "EPIC-Skills":
        train_txt = "train_split" + opts.split + ".txt"
        val_txt = "test_split" + opts.split + ".txt"
        savedir = os.path.join(opts.dataset, opts.task, opts.arg, opts.split, "lap_"+opts.lap)

    # paths dict
    train_list, valid_list, feature_path, writedir, ckptdir, resultdir = make_dirs(args, opts, 
                                                                            train_txt, val_txt, savedir, mode="eval")
    paths = {'train_list': train_list, 'valid_list': valid_list, 'feature_path': feature_path, 
                                 'writedir': writedir, 'ckptdir': ckptdir, 'resultdir': resultdir}


    # ====== Model ======
    # → 2branch(pos and neg)
    if args.disparity_loss and args.rank_aware_loss:
        model = {'p_att': None, 'n_att': None}
    # → 1branch
    else:
        model = {'att': None}
    # attention model
    for k in model.keys():
        model[k] = RAAN(args, uniform=False)
        model[k] = model[k].to(device)


    # ====== Dataloader ======
    evalloader = torch.utils.data.DataLoader(
        SkillDataSet(paths["feature_path"], paths["valid_list"], input_feature=args.input_feature),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

   

    # ====== Eval ======
    Evaluator = Eval_Runner(opts, args, device, paths, model, evalloader)

    Evaluator.evaluate(is_best=True)



if __name__ == '__main__':
    main()


    
