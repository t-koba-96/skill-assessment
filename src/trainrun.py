import os
import time
import torch
import glob
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.util import AverageMeter, data_augmentation, accuracy, sec2str


# ====== Runner class ======

class Train_Runner():

    def __init__(self, opts, args, device, paths, models, dataloaders, criterions, optimizers):

        self.opts = opts
        self.args = args
        self.device = device
        self.dataloaders = dataloaders
        self.models = models
        self.att_branches = self.models["attention"].keys()
        self.criterions = criterions
        self.optimizers = optimizers
        self.resultdir = paths["resultdir"]
        self.writer = SummaryWriter(self.resultdir)

        self.checkpoint_dict = None



    def train_without_uniform(self, epoch):

        # start epoch
        begin = time.time()
        # meters
        Meters = UpdateMeters(self.args)

        # model train mode
        model = self.models["attention"][list(self.att_branches)[0]]
        model.train()
        self.optimizers["phase0"].zero_grad()

        # progress bar 
        with tqdm(total=len(self.dataloaders["train"].dataset)) as pbar:
            pbar.set_description(f"Epoch[{epoch}/{self.args.epochs}](Train)")

            # ====== iter ======  
            for i, (vid_pos, vid_neg, vid_list) in enumerate(self.dataloaders["train"]):
                batch_size = vid_pos.size(0)
                vid_pos_gpu = vid_pos.to(self.device)
                vid_neg_gpu = vid_neg.to(self.device)
                # add gaussian noise
                if self.args.transform:
                    vid_pos_gpu, vid_neg_gpu = data_augmentation(vid_pos_gpu, vid_neg_gpu, self.args, self.device)
                    
                # make target label
                target = torch.ones(batch_size)
                target = target.to(self.device)

                # calc score, attention
                score_pos, att_pos, sp_att_pos = model(vid_pos_gpu)
                score_neg, att_neg, sp_att_neg = model(vid_neg_gpu)

                # mean all filter
                score_pos = score_pos.mean(dim=1)
                score_neg = score_neg.mean(dim=1)

                # measure accuracy
                prec, correct_num = accuracy(score_pos.data, score_neg.data)

                # loss update  
                ranking_loss = 0
                diversity_loss = 0
                all_losses = 0
                # ranking_loss : margin - score_pos + score_neg
                ranking_loss += self.criterions["ranking"](score_pos, score_neg, target)
                all_losses += ranking_loss
                if self.args.diversity_loss:
                    div_loss_att_pos = self.criterions["diversity"](att_pos, self.args, self.device)
                    div_loss_att_neg = self.criterions["diversity"](att_neg, self.args, self.device)
                    diversity_loss += self.args.lambda_param * (div_loss_att_pos + div_loss_att_neg)
                    all_losses += diversity_loss
                    
                # backprop
                all_losses.backward()
                self.optimizers["phase0"].step()
                self.optimizers["phase0"].zero_grad()

                # update records
                records = {"ranking" : ranking_loss.item(), "diversity" : diversity_loss.data.item(), "total_loss" : all_losses.data.item(),
                           "acc" : prec, "correct" : correct_num, "batch_time" : time.time() - begin }
                Meters.update(records, batch_size)

                # measure elapsed time
                begin = time.time()
                total_time = time.time() - self.args.start_time  

                # progress bar
                pbar.update(batch_size)
                pbar.set_postfix({"Loss":"{:.4f}".format(Meters.meters['total_loss'].avg), 
                                  "Acc":"{:.3f}({}/{})".format(Meters.meters['acc'].avg, Meters.meters['correct'].sum, len(self.dataloaders["train"].dataset))})

        # tboard log
        tensorboard_log_train(Meters.meters, 'train', epoch, self.writer) 



    def train_with_uniform(self, epoch, phase=0):

        # start epoch
        begin = time.time()
        # meters
        Meters = UpdateMeters(self.args)
        
        # model train mode
        for k in self.att_branches:
            self.models["attention"][k].train()
        self.models["uniform"].train()
        self.optimizers["phase0"].zero_grad()
        self.optimizers["phase1"].zero_grad()

        # progress bar 
        with tqdm(total=len(self.dataloaders["train"].dataset)) as pbar:
            pbar.set_description(f"Epoch[{epoch}/{self.args.epochs}](Train)")

            # ====== iter ====== 
            for i, (vid_pos, vid_neg, vid_list) in enumerate(self.dataloaders["train"]):
                batch_size = vid_pos.size(0)
                vid_pos_gpu = vid_pos.to(self.device)
                vid_neg_gpu = vid_neg.to(self.device)
                ## add small amount of gaussian noise
                if self.args.transform:
                    vid_pos_gpu, vid_neg_gpu = data_augmentation(vid_pos_gpu, vid_neg_gpu, self.args, self.device)
                    
                # make target label
                target = torch.ones(batch_size)
                target = target.to(self.device)

                # calc score, attention
                all_score_pos, all_score_neg, score_pos, score_neg, att_pos, att_neg = {}, {}, {}, {}, {}, {}
                final_score_pos = torch.zeros(batch_size).to(self.device)
                final_score_neg = torch.zeros(batch_size).to(self.device)
                ### attention model ###
                for k in self.att_branches:
                    all_score_pos[k], att_pos[k] , sp_att_pos = self.models["attention"][k](vid_pos_gpu)
                    all_score_neg[k], att_neg[k] , sp_att_neg = self.models["attention"][k](vid_neg_gpu)
                    # mean all filter
                    score_pos[k] = all_score_pos[k].mean(dim=1)
                    score_neg[k] = all_score_neg[k].mean(dim=1)
                    final_score_pos += score_pos[k].data
                    final_score_neg += score_neg[k].data
                ### uniform model ###
                uniformscore_pos, _ , _ = self.models["uniform"](vid_pos_gpu)
                uniformscore_neg, _ , _ = self.models["uniform"](vid_neg_gpu)
                # mean all filter
                uniformscore_pos = uniformscore_pos.mean(dim=1)
                uniformscore_neg = uniformscore_neg.mean(dim=1)

                # measure accuracy
                prec, correct_num = accuracy(final_score_pos, final_score_neg)
                prec_uniform, correct_num_uniform = accuracy(uniformscore_pos.data, uniformscore_neg.data)

                # loss calc
                ranking_loss = 0
                ranking_loss_uniform = 0
                diversity_loss = 0
                disparity_loss = 0
                rank_aware_loss = 0
                # ranking_loss , disparity_loss
                for k in self.att_branches:
                    ranking_loss += self.criterions["ranking"](score_pos[k], score_neg[k], target) / len(self.att_branches)
                    disparity_loss += self.criterions["disparity"](all_score_pos[k], all_score_neg[k], uniformscore_pos, uniformscore_neg, 
                                                        target, self.args.m2, self.device, self.args.compare_loss_version) / len(self.att_branches)
                ranking_loss_uniform += self.criterions["ranking"](uniformscore_pos, uniformscore_neg, target)
                # rank_aware_loss
                if self.args.rank_aware_loss:
                    rank_aware_loss += self.criterions["disparity"](all_score_pos[list(self.att_branches)[0]], all_score_neg[list(self.att_branches)[1]], uniformscore_pos,
                                                        uniformscore_neg, target, self.args.m3, self.device, self.args.compare_loss_version)
                # diversity_loss
                if self.args.diversity_loss:
                    div_loss_att_pos, div_loss_att_neg = 0, 0
                    for k in self.att_branches:
                        div_loss_att_pos += self.criterions["diversity"](att_pos[k], self.args, self.device)
                        div_loss_att_neg += self.criterions["diversity"](att_neg[k], self.args, self.device)
                    diversity_loss += self.args.lambda_param * (div_loss_att_pos + div_loss_att_neg) / len(self.att_branches)

                # loss update
                all_losses = 0
                # ranking loss
                if phase == 0:
                    all_losses += ranking_loss
                    all_losses += ranking_loss_uniform
                # other 3 losses
                else:
                    all_losses += disparity_loss
                    if self.args.rank_aware_loss:
                        all_losses += rank_aware_loss
                    if self.args.diversity_loss:
                        all_losses += diversity_loss

                # backprop
                all_losses.backward()
                if phase == 0:
                    self.optimizers["phase0"].step()
                    self.optimizers["phase0"].zero_grad()
                    phase = 1
                else:
                    self.optimizers["phase1"].step()
                    self.optimizers["phase1"].zero_grad()
                    phase = 0
        
                # update records
                records = {"ranking" : ranking_loss.item(), "ranking_uniform" : ranking_loss_uniform.item(), 
                           "disparity" : disparity_loss.item(), "rank_aware" : rank_aware_loss.item() , "diversity" : diversity_loss.data.item(), 
                           "total_loss" : all_losses.data.item(), "acc" : prec, "acc_uniform" : prec_uniform,
                           "correct" : correct_num, "batch_time" : time.time() - begin }
                Meters.update(records, batch_size, phase = phase)

                # measure elapsed time
                begin = time.time()
                total_time = time.time() - self.args.start_time

                # progress bar
                pbar.update(batch_size)
                pbar.set_postfix({"Loss":"{:.4f}".format(Meters.meters['total_loss'].avg), 
                                  "Acc":"{:.3f}({}/{})".format(Meters.meters['acc'].avg, Meters.meters['correct'].sum, len(self.dataloaders["train"].dataset))})

        # tboard log
        tensorboard_log_train(Meters.meters, 'train', epoch, self.writer)
        return phase



    def validate(self, epoch):

        # start epoch
        begin = time.time()
        # meters
        Meters = UpdateMeters(self.args)

        # model evaluate mode
        for k in self.att_branches:
            self.models["attention"][k].eval()

        with torch.no_grad():
            # progress bar 
            with tqdm(total=len(self.dataloaders["valid"].dataset)) as pbar:
                pbar.set_description(f"Epoch[{epoch}/{self.args.epochs}](Valid)")

                # ====== iter ====== 
                for i, (vid_pos, vid_neg, vid_list) in enumerate(self.dataloaders["valid"]):
                    batch_size = vid_pos.size(0)
                    vid_pos_gpu = vid_pos.to(self.device)
                    vid_neg_gpu = vid_neg.to(self.device)

                    # make target label
                    target = torch.ones(batch_size)
                    target = target.to(self.device)

                    # calc score, attention
                    all_score_pos, all_score_neg, score_pos, score_neg, att_pos, att_neg = {}, {}, {}, {}, {}, {}
                    final_score_pos = torch.zeros(batch_size).to(self.device)
                    final_score_neg = torch.zeros(batch_size).to(self.device)
                    for k in self.att_branches:
                        all_score_pos[k], att_pos[k], sp_att_pos = self.models["attention"][k](vid_pos_gpu)
                        all_score_neg[k], att_neg[k], sp_att_neg = self.models["attention"][k](vid_neg_gpu)
                        score_pos[k] = all_score_pos[k].mean(dim=1)
                        score_neg[k] = all_score_neg[k].mean(dim=1)
                        final_score_pos += score_pos[k].data
                        final_score_neg += score_neg[k].data

                    # measure accuracy 
                    prec, correct_num = accuracy(final_score_pos, final_score_neg)

                    # loss calc
                    ranking_loss = 0
                    diversity_loss = 0
                    all_losses = 0
                    for k in self.att_branches:
                        ranking_loss += self.criterions["ranking"](score_pos[k], score_neg[k], target) / len(self.att_branches)
                    all_losses += ranking_loss
                    if self.args.diversity_loss:
                        div_loss_att_pos, div_loss_att_neg = 0, 0
                        for k in self.att_branches:
                            div_loss_att_pos += self.criterions["diversity"](att_pos[k], self.args, self.device)
                            div_loss_att_neg += self.criterions["diversity"](att_neg[k], self.args, self.device)
                        diversity_loss += self.args.lambda_param*(div_loss_att_pos + div_loss_att_neg) / len(self.att_branches)
                        all_losses += diversity_loss
                    
                    # update records
                    records = {"ranking" : ranking_loss.item(), "diversity" : diversity_loss.data.item(), "total_loss" : all_losses.data.item(),
                               "acc" : prec, "correct" : correct_num, "batch_time" : time.time() - begin }
                    Meters.update(records, batch_size)

                    # measure elapsed time
                    begin = time.time()

                    # progress bar
                    pbar.update(batch_size)
                    pbar.set_postfix({"Loss":"{:.4f}".format(Meters.meters['total_loss'].avg), 
                                      "Acc":"{:.3f}({}/{})".format(Meters.meters['acc'].avg, Meters.meters['correct'].sum, len(self.dataloaders["valid"].dataset))})

        # tboard log
        tensorboard_log_test(Meters.meters, 'val', epoch, self.writer)
        
        return Meters.meters['acc'].avg


    def record_score(self, epoch=10, is_best=False):

        if is_best:
            weight_path = glob.glob(os.path.join(self.resultdir, 'best_score*'))[0]
        else:
            weight_path = glob.glob(os.path.join(self.resultdir, 'epoch_' + str(epoch).zfill(4) + '*'))[0]
        print('Loading checkpoint file ... {}'.format(weight_path))
        for k in self.att_branches:
            self.models["attention"][k].load_state_dict(torch.load(weight_path)['state_dict_' + k])
        print('Loaded model ... epoch : {:04d}  prec_score : {:.4f}'.format(torch.load(weight_path)["epoch"], 
                                                                            torch.load(weight_path)["prec_score"]))
        self.write_result(torch.load(weight_path)["prec_score"])
        print('Record Score Done!')

        return 
        


    def make_ckpt(self, epoch, prec):

        checkpoint_dict = {'epoch': epoch, 'prec_score': prec}
        for k in self.att_branches:
            checkpoint_dict['state_dict_' + k] = self.models["attention"][k].state_dict()
        if self.args.disparity_loss:
            checkpoint_dict['state_dict_uniform'] = self.models["uniform"].state_dict()

        return checkpoint_dict



    def save_checkpoint(self, epoch, score, ckpt=True, is_best= True):

        state = self.make_ckpt(epoch, score)
        if ckpt:
            savefile = os.path.join(self.resultdir, "epoch_{:04d}_acc_{:.4f}.ckpt".format(epoch, score))
            torch.save(state, savefile)
            print(('Saving checkpoint file ... epoch_{:04d}_acc_{:.4f}.ckpt'.format(epoch, score)))
        if is_best:
            if glob.glob(os.path.join(self.resultdir, 'best_score*')):
                os.remove(glob.glob(os.path.join(self.resultdir, 'best_score*'))[0])
            best_name = os.path.join(self.resultdir, "best_score_epoch_{:04d}_acc_{:.4f}.ckpt".format(epoch, score))
            torch.save(state, best_name)
            print(('Saving checkpoint file ... best_score_epoch_{:04d}_acc_{:.4f}.ckpt'.format(epoch, score)))



    def write_result(self, prec):
        with open(os.path.join(self.opts.result_dir, self.opts.task+"_score_ranking.txt"), mode='a') as a:
            a.write('[prec_score: {}] : [arg : {}]\n'.format(prec, self.opts.arg+"_lap"+self.opts.lap))
        with open(os.path.join(self.opts.result_dir, self.opts.task+"_score_ranking.txt"), mode='r') as f:
            data = f.readlines()
            data = sorted(data, reverse=True)
        with open(os.path.join(self.opts.result_dir, self.opts.task+"_score_ranking.txt"), mode='w') as w:
            for d in data:
                w.write(d)
        print("Updated ... {}".format(self.opts.task+"/scores.txt"))



    def writer_close(self):

        self.writer.close()




# ====== Earlystopping class ======

class earlystopping():
   def __init__(self, patience, init_score):
       self._step = 0
       self._score = init_score
       self.patience = patience

   def validate(self, score):
       if self._score > score:
           self._step += 1
           if self._step == self.patience:
               return True
       else:
           self._step = 0
           self._score = score
       print("Earlystopcount {}".format(self._step)) 
       print("\n")
       return False



# ====== Update meters ======

class UpdateMeters():
    def __init__(self, args):
        self.meters = {'batch_time': AverageMeter(), 'total_loss': AverageMeter(), 
                       'phase0': AverageMeter(), 'phase1': AverageMeter(),
                       'ranking': AverageMeter(), 'ranking_uniform': AverageMeter(),
                       'diversity': AverageMeter(), 'disparity': AverageMeter(),
                       'rank_aware': AverageMeter(), 'correct': AverageMeter(), 
                       'acc': AverageMeter(), 'acc_uniform': AverageMeter()}
        self.meters_list = ['batch_time', 'correct']
        self.batch_meters_list = ['ranking', 'ranking_uniform', 'total_loss', 'disparity', 'acc', 'acc_uniform']
        self.phase_meters_list = ['phase0', 'phase1']
        self.args = args

    def update(self, records, batch_size, phase = -1):
        # udpate record
        for record in records.keys():
            if record in self.meters_list:
                self.meters[record].update(records[record])
            elif record in self.batch_meters_list:
                self.meters[record].update(records[record], batch_size)
            elif record == 'diversity':
                if self.args.diversity_loss:
                    self.meters[record].update(records[record], batch_size)
            elif record == 'rank_aware':
                if self.args.rank_aware_loss:
                    self.meters[record].update(records[record], batch_size)
        # update phase loss
        if phase == 0 or phase == 1:
            self.meters['phase'+str(phase)].reset_val()
            self.meters['phase'+str(1-phase)].update(records["total_loss"], batch_size)



# ====== Tensorboard log ======

def tensorboard_log_test(meters, mode, epoch, writer):
    tb_items = ['_loss/total_loss', '_loss/ranking_loss', '_loss/diversity_loss', '_score/acc']
    meter_items = ['total_loss', 'ranking', 'diversity', 'acc']
    for tb_item, meter_item in zip(tb_items, meter_items):
        writer.add_scalar(mode+tb_item, meters[meter_item].avg, epoch)

def tensorboard_log_train(meters, mode, epoch, writer):
    tensorboard_log_test(meters, mode, epoch, writer)
    tb_items = ['_loss/disparity_loss', '_loss/ranking_loss_uniform', '_score/acc_uniform', 
                '_loss/rank_aware_loss', '_loss/phase0(ranking)_loss', '_loss/phase1(other)_loss']
    meter_items = ['disparity', 'ranking_uniform', 'acc_uniform', 'rank_aware', 'phase0', 'phase1']
    for tb_item, meter_item in zip(tb_items, meter_items):
        writer.add_scalar(mode+tb_item, meters[meter_item].avg, epoch)