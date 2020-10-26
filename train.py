import argparse
import yaml
import torch
import time
import os

from addict import Dict

from main.util import make_dirs
from main.dataset import SkillDataSet
from main.model import RAAN
from main.trainrun import Train_Runner, earlystopping

'''
default == using 'cuda:0'
'''

def get_arguments():

    parser = argparse.ArgumentParser(description='training regression network')
    parser.add_argument('arg', type=str, help='arguments file name')
    parser.add_argument('dataset', type=str, help='choose dataset("EPIC-Skills" or "BEST")')
    parser.add_argument('task', type=str, help='choose task(e.g. "drawing" for EPIC-Skills, "origami" for BEST)')
    parser.add_argument('--lap', type=str, default= '1', help='train lap count(to divide file for same yaml)')
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
    print(('\n''[Options]\n''{0}\n''\n'
           '[Arguements]\n''{1}\n''\n'.format(opts, args)))
    # device setting
    opts.cuda = list(map(str,opts.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(opts.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ====== Path ======
    if opts.dataset == "BEST":
        train_txt = "train.txt"
        val_txt = "test.txt"
        savedir = os.path.join(opts.dataset, opts.task, opts.arg, "lap_"+opts.lap)
    elif opts.dataset == "EPIC-Skills":
        train_txt = "train_split" + opts.split + ".txt"
        val_txt = "test_split" + opts.split + ".txt"
        savedir = os.path.join(opts.dataset, opts.task, opts.arg, opts.split, "lap_"+opts.lap)
    # Paths dict
    paths = make_dirs(args, opts, train_txt, val_txt, savedir)


    # ====== Model ======
    # attention branch
    # 2 branch(pos and neg)
    if args.disparity_loss and args.rank_aware_loss:
        model_attention = {'pos': None, 'neg': None}
    # 1 branch
    else:
        model_attention = {'att': None}
    for k in model_attention.keys():
        model_attention[k] = RAAN(args)
        model_attention[k] = model_attention[k].to(device)
    # uniform branch
    if args.disparity_loss:
        model_uniform = RAAN(args, uniform=True)
        model_uniform = model_uniform.to(device)
    else:
        model_uniform = None
    # Models dict
    models = {"attention" : model_attention , "uniform" : model_uniform}


    # ====== Dataloader ======
    # train_data = train_vid_list.txt 
    train_loader = torch.utils.data.DataLoader(
        SkillDataSet(paths["feature_path"], paths["train_list"], ftr_tmpl='{}_{}.npz'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    # validate_data = test_vid_list.txt
    valid_loader = torch.utils.data.DataLoader(
        SkillDataSet(paths["feature_path"], paths["valid_list"], ftr_tmpl='{}_{}.npz'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    # Dataloader dict
    dataloader = {"train" : train_loader, "valid" : valid_loader}


    # ====== Loss, Optimizer ======
    criterion = torch.nn.MarginRankingLoss(margin=args.m1)
    # optimizer
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
        optimizer_base = torch.optim.Adam(list(model_uniform.parameters()) + model_params, args.lr)
        # optimizer for attention layer
        optimizer_attention = torch.optim.Adam(attention_params, args.lr*0.1)
    # without uniform
    else:
        model = models["attention"][list(models["attention"].keys())[0]]
        optimizer_base = torch.optim.Adam(model.parameters(), args.lr)
        optimizer_attention = None
    # Optimizer dict
    optimizer = {"base" : optimizer_base, "attention" : optimizer_attention}
    


    # ====== Train ======
    Trainer = Train_Runner(args, device, dataloader, models, criterion, optimizer, paths)

    ###  epochs  ###
    best_prec = Trainer.validate(0)
    print("\n")
    early_stop = earlystopping(args.earlystopping, best_prec)
    val_num = 0
    stop_count = 0
    phase = 0

    for epoch in range(args.start_epoch, args.epochs):
        # train
        if args.disparity_loss:
            phase = Trainer.train_with_uniform(epoch, phase=phase)
        else:
            Trainer.train_without_uniform(epoch)
        print('\n')

        # valid
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            # validation
            val_num += 1
            prec = Trainer.validate(epoch+1)
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)

            # save model
            # if ckpt_freq
            if (val_num) % args.ckpt_freq == 0 or epoch == args.epochs - 1:
                Trainer.save_checkpoint(epoch+1, prec, ckpt=True, is_best=is_best)
            # not ckpt_freq but has best_score
            elif is_best:
                Trainer.save_checkpoint(epoch+1, prec, ckpt=False, is_best=is_best)
            
            # early stop
            end_run = early_stop.validate(prec)
            if end_run:
                print("Valid score did not improve for {} rounds ... earlystopping\n".format(args.earlystopping))
                writer.close()
                return

    writer.close()
    return


if __name__ == '__main__':
    main()


    
