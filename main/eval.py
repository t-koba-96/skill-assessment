import argparse
import yaml
import torch
import time
import os
import sys

from addict import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.util import make_dirs
from src.dataset import SkillDataSet
from src.model import RAAN
from src.evalrun import Eval_Runner

'''
default == using 'cuda:0'
'''

def get_arguments():

    parser = argparse.ArgumentParser(description='Evaluation network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('task', type=str, help='choose task[apply_eyeliner | braid_hair | origami | scrambled_egg | tie_tie]')
    parser.add_argument('lap', type=str, help='train lap name(count)')
    parser.add_argument('--epoch', type=str, default='best', help='Model weight [best | epoch_num ] ')
    parser.add_argument('--cuda', type=str, default= [0], help='choose cuda num')
    # Dirs
    parser.add_argument('--root_dir', type=str, default= "results", help='dir for args')
    parser.add_argument('--data_dir', type=str, default= "../../local/dataset/skill", help='dir for dataset')
    parser.add_argument('--result_dir', type=str, default= "results", help='dir for result')
    parser.add_argument('--demo_dir', type=str, default= "demo", help='dir for demo')
    return parser.parse_args()



def main():

    # ====== Args ======
    # start time
    start_time = time.time()

    # parser args
    opts = get_arguments()
    # yaml args
    args = Dict(yaml.safe_load(open(os.path.join(opts.root_dir ,opts.arg, 'arg.yaml'))))
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
    train_list, valid_list, feature_path, resultdir, demodir = make_dirs(args, opts, mode="eval")
    # paths dict
    paths = {'train_list': train_list, 'valid_list': valid_list, 'feature_path': feature_path, 
                                 'resultdir': resultdir, 'demodir': demodir}


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


    
