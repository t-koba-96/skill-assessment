import argparse
import yaml
import torch
import time
import os

from addict import Dict

from main.util import make_dirs
from main.loss import diversity_loss, disparity_loss
from main.dataset import SkillDataSet
from main.model import RAAN
from main.trainrun import Train_Runner, earlystopping

'''
default == using 'cuda:0'
'''

def get_arguments():

    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('task', type=str, help='choose task(e.g. "drawing" for EPIC-Skills, "origami" for BEST)')
    parser.add_argument('lap', type=str, help='train lap name(count)')
    parser.add_argument('--dataset', type=str, default= 'BEST', help='choose dataset [ EPIC-Skills | BEST ] ')
    parser.add_argument('--split', type=str, default= '1', help='Splits for EPIC-Skills')
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
    train_list, valid_list, feature_path, writedir, ckptdir, _ = make_dirs(args, opts, train_txt, val_txt, savedir)
    paths = {'train_list': train_list, 'valid_list': valid_list, 
             'feature_path': feature_path, 'writedir': writedir, 'ckptdir': ckptdir}



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


    
