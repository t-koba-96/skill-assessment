import os
import time
import torch
import glob
from torch.utils.tensorboard import SummaryWriter

from main.loss import diversity_loss, multi_rank_loss
from main.util import AverageMeter, data_augmentation, accuracy, sec2str


# ====== Runner class ======

class Train_Runner():

    def __init__(self, args, device, dataloader, models, criterion, optimizer, paths):

        self.args = args
        self.device = device
        self.dataloader = dataloader
        self.models = models
        self.criterion = criterion
        self.optimizer = optimizer
        self.ckptdir = paths["ckptdir"]
        self.writer = SummaryWriter(paths["writedir"])

        self.checkpoint_dict = None



    def train_without_uniform(self, epoch):

        begin = time.time()
        print('Training : Epoch[{}/{}]'.format(epoch+1, self.args.epochs))

        av_meters = {'batch_time': AverageMeter(), 'losses': AverageMeter(),
                    'phase0_loss': AverageMeter(), 'phase1_loss': AverageMeter(),
                    'ranking_losses': AverageMeter(), 'diversity_losses': AverageMeter(), 
                    'correct': AverageMeter(), 'acc': AverageMeter()}
        
        model = self.models["attention"][list(self.models["attention"].keys())[0]]
        model.train()
        
        self.optimizer["base"].zero_grad()

        ## iter ## 
        for i, (input1, input2, vid_list) in enumerate(self.dataloader["train"]):
            input1_gpu = input1.to(self.device)
            input2_gpu = input2.to(self.device)
            ## add small amount of gaussian noise
            if self.args.transform:
                input1_gpu, input2_gpu = data_augmentation(input1_gpu, input2_gpu, self.device)
                
            labels = torch.ones(input1.size(0))
            target = labels.to(self.device)

            output1, att1 = model(input1_gpu)
            output2, att2 = model(input2_gpu)
            # mean all filter
            output1 = output1.mean(dim=1)
            output2 = output2.mean(dim=1)

            
            ranking_loss = 0
            all_losses = 0
            # ranking_loss : margin - output1 + output2
            ranking_loss += self.criterion(output1, output2, target)
            all_losses += ranking_loss
            if self.args.diversity_loss:
                div_loss_att1 = diversity_loss(att1, self.args, self.device)
                div_loss_att2 = diversity_loss(att2, self.args, self.device)
                all_losses += self.args.lambda_param*(div_loss_att1 + div_loss_att2)
                
            # measure accuracy and backprop
            prec, cor = accuracy(output1.data, output2.data)

            all_losses.backward()

            self.optimizer["base"].step()
            self.optimizer["base"].zero_grad()

            # record losses
            av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0))
            if self.args.diversity_loss:
                av_meters['diversity_losses'].update(self.args.lambda_param*(div_loss_att1.item()+div_loss_att2.item()),
                                                    input1.size(0))
            av_meters['phase0_loss'].update(all_losses.data.item(), input1.size(0))
            av_meters['losses'].update(all_losses.data.item(), input1.size(0))
            av_meters['acc'].update(prec, input1.size(0))
            av_meters['correct'].update(cor)

            # measure elapsed time
            av_meters['batch_time'].update(time.time() - begin)
            begin = time.time()
            total_time = time.time() - self.args.start_time

            if i % (self.args.print_freq) == 0:
                console_log_train_batch(av_meters, epoch+1, self.args.epochs, i+1, 
                                        len(self.dataloader["train"]), total_time, input1.size(0))

        console_log_train(av_meters, epoch+1, self.args.epochs, total_time)
        tensorboard_log(av_meters, 'train', epoch+1, self.writer) 



    def train_with_uniform(self, epoch, phase=0):

        begin = time.time()
        print('Training : Epoch[{}/{}]'.format(epoch+1, self.args.epochs))

        av_meters = {'batch_time': AverageMeter(), 'losses': AverageMeter(),
                    'phase0_loss': AverageMeter(), 'phase1_loss': AverageMeter(),
                    'ranking_losses': AverageMeter(), 'ranking_losses_uniform': AverageMeter(),
                    'diversity_losses': AverageMeter(), 'disparity_losses': AverageMeter(),
                    'rank_aware_losses': AverageMeter(), 'correct': AverageMeter(), 
                    'acc': AverageMeter(), 'acc_uniform': AverageMeter()}
        
        for k in self.models["attention"].keys():
            self.models["attention"][k].train()
        self.models["uniform"].train()
        
        self.optimizer["base"].zero_grad()
        self.optimizer["attention"].zero_grad()

        ## iter ## 
        for i, (input1, input2, vid_list) in enumerate(self.dataloader["train"]):
            input1_gpu = input1.to(self.device)
            input2_gpu = input2.to(self.device)
            ## add small amount of gaussian noise
            if self.args.transform:
                input1_gpu, input2_gpu = data_augmentation(input1_gpu, input2_gpu, self.device)
                
            labels = torch.ones(input1.size(0))
            target = labels.to(self.device)

            all_output1, all_output2, output1, output2, att1, att2 = {}, {}, {}, {}, {}, {}
            # attention model
            for k in self.models["attention"].keys():
                # batch * filter 
                all_output1[k], att1[k] = self.models["attention"][k](input1_gpu)
                all_output2[k], att2[k] = self.models["attention"][k](input2_gpu)
                # mean all filter
                output1[k] = all_output1[k].mean(dim=1)
                output2[k] = all_output2[k].mean(dim=1)
            # uniform model
            output1_uniform, _ = self.models["uniform"](input1_gpu)
            output2_uniform, _ = self.models["uniform"](input2_gpu)
            output1_uniform = output1_uniform.mean(dim=1)
            output2_uniform = output2_uniform.mean(dim=1)

            ranking_loss = 0
            ranking_loss_uniform = 0
            disparity_loss = 0
            rank_aware_loss = 0
            for k in self.models["attention"].keys():
                ranking_loss += self.criterion(output1[k], output2[k], target)
                disparity_loss += multi_rank_loss(all_output1[k], all_output2[k], output1_uniform,
                                                    output2_uniform, target, self.args.m2, self.device, self.args.disparity_loss)
            ranking_loss_uniform += self.criterion(output1_uniform, output2_uniform, target)
            if self.args.rank_aware_loss:
                rank_aware_loss += multi_rank_loss(all_output1['pos'], all_output2['neg'], output1_uniform,
                                                    output2_uniform, target, self.args.m3, self.device, self.args.rank_aware_loss)

            if self.args.diversity_loss:
                div_loss_att1, div_loss_att2 = 0, 0
                for k in self.models["attention"].keys():
                    div_loss_att1 += diversity_loss(att1[k], self.args, self.device)
                    div_loss_att2 += diversity_loss(att2[k], self.args, self.device)

            all_losses = 0
            # ranking loss
            if phase == 0:
                all_losses += ranking_loss
                all_losses += ranking_loss_uniform
                av_meters['phase0_loss'].update(all_losses.item(), input1.size(0))
                av_meters['phase1_loss'].reset_val()
            # other 3 losses
            else:
                all_losses += disparity_loss
                if self.args.rank_aware_loss:
                    all_losses += rank_aware_loss
                if self.args.diversity_loss:
                    all_losses += self.args.lambda_param*(div_loss_att1 + div_loss_att2)
                av_meters['phase1_loss'].update(all_losses.item(), input1.size(0))
                av_meters['phase0_loss'].reset_val()


            # final output (e.g. pos + neg)
            output1_all = torch.zeros(output1[list(self.models["attention"].keys())[0]].data.shape)
            output1_all = output1_all.to(self.device)
            output2_all = torch.zeros(output2[list(self.models["attention"].keys())[0]].data.shape)
            output2_all = output2_all.to(self.device)
            for k in self.models["attention"].keys():
                output1_all += output1[k].data
                output2_all += output2[k].data

            # measure accuracy and backprop
            prec, cor = accuracy(output1_all, output2_all)
            prec_uniform, cor_uniform = accuracy(output1_uniform.data, output2_uniform.data)

            all_losses.backward()
            
            # train optimizer alternately
            if phase == 0:
                self.optimizer["base"].step()
                self.optimizer["base"].zero_grad()
                phase = 1
            else:
                self.optimizer["attention"].step()
                self.optimizer["attention"].zero_grad()
                phase = 0
    
            # record losses
            av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0)*len(self.models["attention"].keys()))
            av_meters['ranking_losses_uniform'].update(ranking_loss_uniform.item(), input1.size(0))
            av_meters['disparity_losses'].update(disparity_loss.item(), input1.size(0)*len(self.models["attention"].keys()))
            if self.args.diversity_loss:
                av_meters['diversity_losses'].update(self.args.lambda_param*(div_loss_att1.item()+div_loss_att2.item()),
                                                    input1.size(0)*len(self.models["attention"].keys()))
            if self.args.rank_aware_loss:
                av_meters['rank_aware_losses'].update(rank_aware_loss.item(), input1.size(0))
            av_meters['losses'].update(all_losses.data.item(), input1.size(0))
            av_meters['acc'].update(prec, input1.size(0))
            av_meters['correct'].update(cor)
            av_meters['acc_uniform'].update(prec_uniform, input1.size(0))

            # measure elapsed time
            av_meters['batch_time'].update(time.time() - begin)
            begin = time.time()
            total_time = time.time() - self.args.start_time

            if i % (self.args.print_freq) == 0:
                console_log_train_batch(av_meters, epoch+1, self.args.epochs, i+1, 
                                        len(self.dataloader["train"]), total_time, input1.size(0))

        console_log_train(av_meters, epoch+1, self.args.epochs, total_time)
        tensorboard_log_with_uniform(av_meters, 'train', epoch+1, self.writer)
        return phase



    def validate(self, epoch):

        begin = time.time()
        print('Testing : Epoch[{}/{}]'.format(epoch, self.args.epochs))

        av_meters = {'batch_time': AverageMeter(), 'losses': AverageMeter(),
                    'ranking_losses': AverageMeter(), 'diversity_losses': AverageMeter(),
                    'correct': AverageMeter(), 'acc': AverageMeter()}

        # switch to evaluate mode
        for k in self.models["attention"].keys():
            self.models["attention"][k].eval()


        for i, (input1, input2, vid_list) in enumerate(self.dataloader["valid"]):
            input1_gpu = input1.to(self.device)
            input2_gpu = input2.to(self.device)

            labels = torch.ones(input1.size(0))
            target = labels.to(self.device)

            all_output1, all_output2, output1, output2, att1, att2 = {}, {}, {}, {}, {}, {}
            for k in self.models["attention"].keys():
                all_output1[k], att1[k] = self.models["attention"][k](input1_gpu)
                all_output2[k], att2[k] = self.models["attention"][k](input2_gpu)
                output1[k] = all_output1[k].mean(dim=1)
                output2[k] = all_output2[k].mean(dim=1)

            ranking_loss = 0
            for k in self.models["attention"].keys():
                ranking_loss += self.criterion(output1[k], output2[k], target)
            all_losses = ranking_loss.item()
            if self.args.diversity_loss:
                div_loss_att1, div_loss_att2 = 0, 0
                for k in self.models["attention"].keys():
                    div_loss_att1 += diversity_loss(att1[k], self.args, self.device)
                    div_loss_att2 += diversity_loss(att2[k], self.args, self.device)
                all_losses += self.args.lambda_param*(div_loss_att1 + div_loss_att2)
            
            # final output
            output1_all = torch.zeros(output1[list(self.models["attention"].keys())[0]].data.shape)
            output1_all = output1_all.to(self.device)
            output2_all = torch.zeros(output2[list(self.models["attention"].keys())[0]].data.shape)
            output2_all = output2_all.to(self.device)
            for k in self.models["attention"].keys():
                output1_all += output1[k].data
                output2_all += output2[k].data

            # measure accuracy 
            prec, cor = accuracy(output1_all, output2_all)

            # record losses
            av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0)*len(self.models["attention"].keys()))
            if self.args.diversity_loss:
                av_meters['diversity_losses'].update(self.args.lambda_param*(div_loss_att1.item()+div_loss_att2.item()),
                                                    input1.size(0)*len(self.models["attention"].keys()))
            av_meters['losses'].update(all_losses.data.item(), input1.size(0))
            av_meters['acc'].update(prec, input1.size(0))
            av_meters['correct'].update(cor)

            # measure elapsed time
            av_meters['batch_time'].update(time.time() - begin)
            begin = time.time()

        console_log_test(av_meters)
        tensorboard_log(av_meters, 'val', epoch, self.writer)
        
        return av_meters['acc'].avg



    def make_ckpt(self, epoch, prec):

        checkpoint_dict = {'epoch': epoch, 'prec_score': prec}
        for k in self.models["attention"].keys():
            checkpoint_dict['state_dict_' + k] = self.models["attention"][k].state_dict()
        if self.args.disparity_loss:
            checkpoint_dict['state_dict_uniform'] = self.models["uniform"].state_dict()

        return checkpoint_dict



    def save_checkpoint(self, epoch, score, ckpt=True, is_best= True):

        state = self.make_ckpt(epoch, score)
        if ckpt:
            savefile = os.path.join(self.ckptdir, "epoch_{:04d}_acc_{:.4f}.ckpt".format(epoch, score))
            torch.save(state, savefile)
            print(('Saving checkpoint file ... epoch_{:04d}_acc_{:.4f}.ckpt'.format(epoch, score)))
        if is_best:
            if glob.glob(os.path.join(self.ckptdir, 'best_score*')):
                os.remove(glob.glob(os.path.join(self.ckptdir, 'best_score*'))[0])
            best_name = os.path.join(self.ckptdir, "best_score_epoch_{:04d}_acc_{:.4f}.ckpt".format(epoch, score))
            torch.save(state, best_name)
            print(('Saving checkpoint file ... best_score_epoch_{:04d}_acc_{:.4f}.ckpt'.format(epoch, score)))




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


# ====== Console log ======

def console_log_train_batch(av_meters, epoch, total_epoch, iter, 
                            loader_len, total_time, total):
    print(('Epoch:[{0}/{1}]Iter:[{2}/{3}]\t'
           'Time:{batch_time.val:.2f}s({4})\t\t'
           'Loss(Rank):{phase0_loss.val:.5f}\t'
           'Loss(Others):{phase1_loss.val:.5f}\t'
           'Accuracy:{acc.val:.3f}({correct.val}/{5})'.format(
               epoch, total_epoch, iter, loader_len, 
               sec2str(total_time), total, 
               batch_time=av_meters['batch_time'], loss=av_meters['losses'], 
               phase0_loss=av_meters['phase0_loss'], phase1_loss=av_meters['phase1_loss'],
               correct=av_meters['correct'], acc=av_meters['acc'])))

def console_log_train(av_meters, epoch, total_epoch, total_time):
    print(('[Epoch:({0}/{1}]) Results]\t'
           'Time:{batch_time.sum:.2f}s({2})\t\t'
           'Loss(Rank):{phase0_loss.avg:.5f}\t'
           'Loss(Others):{phase1_loss.avg:.5f}\t'
           'Accuracy:{acc.avg:.3f}({correct.sum}/{acc.count})'.format(
               epoch, total_epoch, sec2str(total_time), 
               batch_time=av_meters['batch_time'], loss=av_meters['losses'], 
               phase0_loss=av_meters['phase0_loss'], phase1_loss=av_meters['phase1_loss'],
               correct=av_meters['correct'], acc=av_meters['acc'])))

def console_log_test(av_meters):
    print(('[Testing Results]\t'
           'Time:{batch_time.sum:.3f}s\t'
           'Loss(Rank):{r_loss.avg:.5f}\t'
           'Loss(Rank+Div):{loss.avg:.5f}\t'
           'Accuracy:{acc.avg:.3f}({correct.sum}/{acc.count})'.format(
               batch_time=av_meters['batch_time'], r_loss=av_meters['ranking_losses'],
               loss=av_meters['losses'],correct=av_meters['correct'], acc=av_meters['acc'])))


# ====== Tensorboard log ======

def tensorboard_log(av_meters, mode, epoch, writer):
    writer.add_scalar(mode+'/total_loss', av_meters['losses'].avg, epoch)
    writer.add_scalar(mode+'/ranking_loss', av_meters['ranking_losses'].avg, epoch)
    writer.add_scalar(mode+'/diversity_loss', av_meters['diversity_losses'].avg, epoch)
    writer.add_scalar(mode+'/acc', av_meters['acc'].avg, epoch)

def tensorboard_log_with_uniform(av_meters, mode, epoch, writer):
    tensorboard_log(av_meters, mode, epoch, writer)
    writer.add_scalar(mode+'/disparity_loss', av_meters['disparity_losses'].avg, epoch)
    writer.add_scalar(mode+'/ranking_loss_uniform', av_meters['ranking_losses_uniform'].avg, epoch)
    writer.add_scalar(mode+'/acc_uniform', av_meters['acc_uniform'].avg, epoch)
    writer.add_scalar(mode+'/rank_aware_loss', av_meters['rank_aware_losses'].avg, epoch)
    writer.add_scalar(mode+'/phase0_loss', av_meters['phase0_loss'].avg, epoch)
    writer.add_scalar(mode+'/phase1_loss', av_meters['phase1_loss'].avg, epoch)
    