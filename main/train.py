import argparse
import yaml
import torch
import time
import os
import sys

from addict import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.util import make_dirs
from src.loss import diversity_loss, disparity_loss
from src.dataset import SkillDataSet
from src.model import RAAN
from src.trainrun import Train_Runner, earlystopping

'''
default == using 'cuda:0'
'''

def get_arguments():

    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('task', type=str, help='choose task[apply_eyeliner | braid_hair | origami | scrambled_egg | tie_tie]')
    parser.add_argument('lap', type=str, help='train lap name(count)')
    parser.add_argument('--cuda', type=str, default= [0], help='choose cuda num')
    # Dirs
    parser.add_argument('--root_dir', type=str, default= "results", help='dir for args')
    parser.add_argument('--data_dir', type=str, default= "../../local/dataset/skill", help='dir for dataset')
    parser.add_argument('--result_dir', type=str, default= "results", help='dir for result')
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
    train_list, valid_list, feature_path, resultdir, _ = make_dirs(args, opts)
    # paths dict
    paths = {'train_list': train_list, 'valid_list': valid_list, 
             'feature_path': feature_path, 'resultdir': resultdir}



    # ====== Models ======
    ### attention branch ###
    # → 2branch(pos and neg)
    if args.disparity_loss and args.rank_aware_loss:
        model_attention = {'p_att': None, 'n_att': None}
    # → 1branch
    else:
        model_attention = {'att': None}
    # attention model
    for k in model_attention.keys():
        model_attention[k] = RAAN(args, uniform=False)
        model_attention[k] = model_attention[k].to(device)

    ### uniform branch ###
    if args.disparity_loss:
        model_uniform = RAAN(args, uniform=True)
        model_uniform = model_uniform.to(device)
    else:
        model_uniform = None

    # models dict
    models = {"attention" : model_attention , "uniform" : model_uniform}



    # ====== Dataloaders ======
    # train_data = train_vid_list.txt 
    train_loader = torch.utils.data.DataLoader(
        SkillDataSet(paths["feature_path"], paths["train_list"], input_feature=args.input_feature),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    # validation_data = test_vid_list.txt
    valid_loader = torch.utils.data.DataLoader(
        SkillDataSet(paths["feature_path"], paths["valid_list"], input_feature=args.input_feature),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # dataloaders dict
    dataloaders = {"train" : train_loader, "valid" : valid_loader}



    # ====== Losses ======
    # lossses
    ranking_loss = torch.nn.MarginRankingLoss(margin=args.m1)

    # criterions dict
    criterions = {"ranking" : ranking_loss, "disparity" : disparity_loss, "diversity" : diversity_loss}



    # ====== Optimizers ======
    # with uniform
    if args.disparity_loss:
        attention_params = []
        model_params = []
        for model in models["attention"].values():
            for name, param in model.named_parameters():
                if param.requires_grad and 'att' in name:
                    attention_params.append(param)
                else:
                    model_params.append(param)
        optimizer_phase0 = torch.optim.Adam(list(model_uniform.parameters()) + model_params, args.lr)
        # optimizer for attention layer
        optimizer_phase1 = torch.optim.Adam(attention_params, args.lr*0.1)
    # without uniform
    else:
        model = models["attention"][list(models["attention"].keys())[0]]
        optimizer_phase0 = torch.optim.Adam(model.parameters(), args.lr)
        optimizer_phase1 = None

    # optimizers dict
    optimizers = {"phase0" : optimizer_phase0, "phase1" : optimizer_phase1}
    


    # ====== Train ======
    Trainer = Train_Runner(opts, args, device, paths, models, dataloaders, criterions, optimizers)

    ###  epochs  ###
    best_prec = Trainer.validate(args.start_epoch-1)
    Trainer.save_checkpoint(0, best_prec, ckpt=False, is_best=True)
    print("\n")
    early_stop = earlystopping(args.earlystopping, best_prec)
    val_num = 0
    stop_count = 0
    phase = 0

    for epoch in range(args.start_epoch, args.epochs+1):
        # train
        if args.disparity_loss:
            phase = Trainer.train_with_uniform(epoch, phase=phase)
        else:
            Trainer.train_without_uniform(epoch)

        # valid
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            # validation
            val_num += 1
            prec = Trainer.validate(epoch)
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)

            # save model
            # if ckpt_freq
            if val_num % args.ckpt_freq == 0 or epoch == args.epochs:
                Trainer.save_checkpoint(epoch, prec, ckpt=True, is_best=is_best)
            # not ckpt_freq but has best_score
            elif is_best:
                Trainer.save_checkpoint(epoch, prec, ckpt=False, is_best=True)
            
            # early stop
            end_run = early_stop.validate(prec)
            if end_run:
                print("Valid score did not improve for {} rounds ... earlystopping\n".format(args.earlystopping))
                Trainer.record_score(is_best=True)
                Trainer.writer_close()
                return

    Trainer.record_score(is_best=True)
    Trainer.writer_close()
    return


if __name__ == '__main__':
    main()


    
