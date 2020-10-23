import argparse
import yaml
import torch
import time
import os

from main import loss, util
from main.util import AverageMeter
from main.dataset import SkillDataSet
from main.model import RAAN

from addict import Dict


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
    # args
    global args, best_prec, writer, device, start_time
    best_prec = 0
    start_time = time.time()
    opts = get_arguments()
    args = Dict(yaml.safe_load(open(os.path.join('args',opts.arg+'.yaml'))))
    print(('\n''[Options]\n''{0}\n''\n'
           '[Arguements]\n''{1}\n''\n'.format(opts, args)))
    opts.cuda = list(map(str,opts.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(opts.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # path setting
    if opts.dataset == "BEST":
        train_txt = "train.txt"
        val_txt = "test.txt"
        savedir = os.path.join(opts.dataset, opts.task, opts.arg, "lap_"+opts.lap)
    elif opts.dataset == "EPIC-Skills":
        train_txt = "train_split" + opts.split + ".txt"
        val_txt = "test_split" + opts.split + ".txt"
        savedir = os.path.join(opts.dataset, opts.task, opts.arg, opts.split, "lap_"+opts.lap)
    train_list, val_list, feat_path, writer, ckptdir\
                    = util.make_dirs(args, opts, train_txt, val_txt, savedir)


    # loading model
    # attention branch
    if args.disparity_loss and args.rank_aware_loss:
        num_attention_branches = 2
        models = {'pos': None, 'neg': None}
    else:
        num_attention_branches = 1
        models = {'att': None}
    for k in models.keys():
        models[k] = RAAN(args)
        models[k] = models[k].to(device)
    # uniform branch
    if args.disparity_loss:
        model_uniform = RAAN(args, uniform=True)
        model_uniform = model_uniform.to(device)


    # dataloader
    # train_data = train_vid_list.txt 
    train_loader = torch.utils.data.DataLoader(
        SkillDataSet(feat_path, train_list, ftr_tmpl='{}_{}.npz'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    # validate_data = test_vid_list.txt
    val_loader = torch.utils.data.DataLoader(
        SkillDataSet(feat_path, val_list, ftr_tmpl='{}_{}.npz'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)


    # loss
    criterion = torch.nn.MarginRankingLoss(margin=args.m1)
    # optimizer
    # with uniform
    if args.disparity_loss:
        attention_params = []
        model_params = []
        for model in models.values():
            for name, param in model.named_parameters():
                if param.requires_grad and 'att' in name:
                    attention_params.append(param)
                else:
                    model_params.append(param)
        optimizer = torch.optim.Adam(list(model_uniform.parameters()) + model_params, args.lr)
        # optimizer for attention layer
        optimizer_attention = torch.optim.Adam(attention_params, args.lr*0.1)
    # without uniform
    else:
        model = models[list(models.keys())[0]]
        optimizer = torch.optim.Adam(model.parameters(), args.lr)


    ###  epochs  ###
    if args.evaluate:
        validate(val_loader, models, criterion, 0)
        print('\n')
    phase = 0
    for epoch in range(args.start_epoch, args.epochs):
        # train
        if args.disparity_loss:
            phase = train_with_uniform(train_loader, models, model_uniform, criterion,
                                       optimizer, optimizer_attention,
                                       epoch, phase=phase)
        else:
            train_without_uniform(train_loader, models, criterion, optimizer, epoch)
        print('\n')

        # validate , save
        validated = False
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec, validated = validate(val_loader, models, criterion, epoch+1)
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
        if (epoch + 1) % args.ckpt_freq == 0 or epoch == args.epochs - 1:
            if not validated:
                prec, validated = validate(val_loader, models, criterion, epoch+1, False)
                is_best = prec > best_prec
                best_prec = max(prec, best_prec)
            checkpoint_dict = {'epoch': epoch + 1, 'best_prec': best_prec}
            for k in models.keys():
                checkpoint_dict['state_dict_' + k] = models[k].state_dict(),
            if args.disparity_loss:
                checkpoint_dict['state_dict_uniform'] = model_uniform.state_dict(),
            util.save_checkpoint(checkpoint_dict, ckptdir, epoch + 1, prec, is_best)
        if validated:
            print('\n')
    writer.close()



def train_without_uniform(train_loader, models, criterion, optimizer, epoch, shuffle=True, phase=0):
    av_meters = {'batch_time': AverageMeter(), 'losses': AverageMeter(),
                 'phase0_loss': AverageMeter(), 'phase1_loss': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'diversity_losses': AverageMeter(), 
                 'correct': AverageMeter(), 'acc': AverageMeter()}
    
    model = models[list(models.keys())[0]]
    model.train()

    begin = time.time()
    print('Training : Epoch[{}/{}]'.format(epoch+1, args.epochs))
    
    optimizer.zero_grad()
    for i, (input1, input2, vid_list) in enumerate(train_loader):
        input1_gpu = input1.to(device)
        input2_gpu = input2.to(device)
        ## add small amount of gaussian noise
        if args.transform:
            input1_gpu, input2_gpu = util.data_augmentation(input1_gpu, input2_gpu, device)
            
        labels = torch.ones(input1.size(0))
        target = labels.to(device)

        output1, att1 = model(input1_gpu)
        output2, att2 = model(input2_gpu)
        # mean all filter
        output1 = output1.mean(dim=1)
        output2 = output2.mean(dim=1)

        
        ranking_loss = 0
        all_losses = 0
        # ranking_loss : margin - output1 + output2
        ranking_loss += criterion(output1, output2, target)
        all_losses += ranking_loss
        if args.diversity_loss:
            div_loss_att1 = loss.diversity_loss(att1, args, device)
            div_loss_att2 = loss.diversity_loss(att2, args, device)
            all_losses += args.lambda_param*(div_loss_att1 + div_loss_att2)
            
        # measure accuracy and backprop
        prec, cor = util.accuracy(output1.data, output2.data)

        all_losses.backward()

        optimizer.step()
        optimizer.zero_grad()

        # record losses
        av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0))
        if args.diversity_loss:
            av_meters['diversity_losses'].update(args.lambda_param*(div_loss_att1.item()+div_loss_att2.item()),
                                                input1.size(0))
        av_meters['phase0_loss'].update(all_losses.data.item(), input1.size(0))
        av_meters['losses'].update(all_losses.data.item(), input1.size(0))
        av_meters['acc'].update(prec, input1.size(0))
        av_meters['correct'].update(cor)

        # measure elapsed time
        av_meters['batch_time'].update(time.time() - begin)
        begin = time.time()
        total_time = time.time() - start_time

        if i % (args.print_freq) == 0:
            console_log_train_batch(av_meters, epoch+1, args.epochs, i+1, 
                                    len(train_loader), total_time, input1.size(0))

    console_log_train(av_meters, epoch+1, args.epochs, total_time)
    tensorboard_log(av_meters, 'train', epoch+1) 



def train_with_uniform(train_loader, models, model_uniform, criterion, optimizer, optimizer_attention, epoch, shuffle=True, phase=0):
    av_meters = {'batch_time': AverageMeter(), 'losses': AverageMeter(),
                 'phase0_loss': AverageMeter(), 'phase1_loss': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'ranking_losses_uniform': AverageMeter(),
                 'diversity_losses': AverageMeter(), 'disparity_losses': AverageMeter(),
                 'rank_aware_losses': AverageMeter(), 'correct': AverageMeter(), 
                 'acc': AverageMeter(), 'acc_uniform': AverageMeter()}
    
    for k in models.keys():
        models[k].train()
    model_uniform.train()
    
    begin = time.time()
    print('Training : Epoch[{}/{}]'.format(epoch+1, args.epochs))
    
    optimizer.zero_grad()
    for i, (input1, input2, vid_list) in enumerate(train_loader):
        input1_gpu = input1.to(device)
        input2_gpu = input2.to(device)
        ## add small amount of gaussian noise
        if args.transform:
            input1_gpu, input2_gpu = util.data_augmentation(input1_gpu, input2_gpu, device)
            
        labels = torch.ones(input1.size(0))
        target = labels.to(device)

        all_output1, all_output2, output1, output2, att1, att2 = {}, {}, {}, {}, {}, {}
        # attention model
        for k in models.keys():
            # batch * filter 
            all_output1[k], att1[k] = models[k](input1_gpu)
            all_output2[k], att2[k] = models[k](input2_gpu)
            # mean all filter
            output1[k] = all_output1[k].mean(dim=1)
            output2[k] = all_output2[k].mean(dim=1)
        # uniform model
        output1_uniform, _ = model_uniform(input1_gpu)
        output2_uniform, _ = model_uniform(input2_gpu)
        output1_uniform = output1_uniform.mean(dim=1)
        output2_uniform = output2_uniform.mean(dim=1)

        ranking_loss = 0
        ranking_loss_uniform = 0
        disparity_loss = 0
        rank_aware_loss = 0
        for k in models.keys():
            ranking_loss += criterion(output1[k], output2[k], target)
            disparity_loss += loss.multi_rank_loss(all_output1[k], all_output2[k], output1_uniform,
                                                output2_uniform, target, args.m2, device, args.disparity_loss)
        ranking_loss_uniform += criterion(output1_uniform, output2_uniform, target)
        if args.rank_aware_loss:
            rank_aware_loss += loss.multi_rank_loss(all_output1['pos'], all_output2['neg'], output1_uniform,
                                                output2_uniform, target, args.m3, device, args.rank_aware_loss)

        if args.diversity_loss:
            div_loss_att1, div_loss_att2 = 0, 0
            for k in models.keys():
                div_loss_att1 += loss.diversity_loss(att1[k], args, device)
                div_loss_att2 += loss.diversity_loss(att2[k], args, device)

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
            if args.rank_aware_loss:
                all_losses += rank_aware_loss
            if args.diversity_loss:
                all_losses += args.lambda_param*(div_loss_att1 + div_loss_att2)
            av_meters['phase1_loss'].update(all_losses.item(), input1.size(0))
            av_meters['phase0_loss'].reset_val()


        # final output (e.g. pos + neg)
        output1_all = torch.zeros(output1[list(models.keys())[0]].data.shape)
        output1_all = output1_all.to(device)
        output2_all = torch.zeros(output2[list(models.keys())[0]].data.shape)
        output2_all = output2_all.to(device)
        for k in models.keys():
            output1_all += output1[k].data
            output2_all += output2[k].data

        # measure accuracy and backprop
        prec, cor = util.accuracy(output1_all, output2_all)
        prec_uniform, cor_uniform = util.accuracy(output1_uniform.data, output2_uniform.data)

        all_losses.backward()
        
        # train optimizer alternately
        if phase == 0:
            optimizer.step()
            optimizer.zero_grad()
            phase = 1
        else:
            optimizer_attention.step()
            optimizer_attention.zero_grad()
            phase = 0
 
        # record losses
        av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0)*len(models.keys()))
        av_meters['ranking_losses_uniform'].update(ranking_loss_uniform.item(), input1.size(0))
        av_meters['disparity_losses'].update(disparity_loss.item(), input1.size(0)*len(models.keys()))
        if args.diversity_loss:
            av_meters['diversity_losses'].update(args.lambda_param*(div_loss_att1.item()+div_loss_att2.item()),
                                                input1.size(0)*len(models.keys()))
        if args.rank_aware_loss:
            av_meters['rank_aware_losses'].update(rank_aware_loss.item(), input1.size(0))
        av_meters['losses'].update(all_losses.data.item(), input1.size(0))
        av_meters['acc'].update(prec, input1.size(0))
        av_meters['correct'].update(cor)
        av_meters['acc_uniform'].update(prec_uniform, input1.size(0))

        # measure elapsed time
        av_meters['batch_time'].update(time.time() - begin)
        begin = time.time()
        total_time = time.time() - start_time

        if i % (args.print_freq) == 0:
            console_log_train_batch(av_meters, epoch+1, args.epochs, i+1, 
                                    len(train_loader), total_time, input1.size(0))

    console_log_train(av_meters, epoch+1, args.epochs, total_time)
    tensorboard_log_with_uniform(av_meters, 'train', epoch+1)
    return phase



def validate(val_loader, models, criterion, epoch, log = True):
    av_meters = {'batch_time': AverageMeter(), 'losses': AverageMeter(),
                 'ranking_losses': AverageMeter(), 'diversity_losses': AverageMeter(),
                 'correct': AverageMeter(), 'acc': AverageMeter()}

    # switch to evaluate mode
    for k in models.keys():
        models[k].eval()

    begin = time.time()
    print('Testing : Epoch[{}/{}]'.format(epoch, args.epochs))

    for i, (input1, input2, vid_list) in enumerate(val_loader):
        input1_gpu = input1.to(device)
        input2_gpu = input2.to(device)

        labels = torch.ones(input1.size(0))
        target = labels.to(device)

        all_output1, all_output2, output1, output2, att1, att2 = {}, {}, {}, {}, {}, {}
        for k in models.keys():
            all_output1[k], att1[k] = models[k](input1_gpu)
            all_output2[k], att2[k] = models[k](input2_gpu)
            output1[k] = all_output1[k].mean(dim=1)
            output2[k] = all_output2[k].mean(dim=1)

        ranking_loss = 0
        for k in models.keys():
            ranking_loss += criterion(output1[k], output2[k], target)
        all_losses = ranking_loss.item()
        if args.diversity_loss:
            div_loss_att1, div_loss_att2 = 0, 0
            for k in models.keys():
                div_loss_att1 += loss.diversity_loss(att1[k], args, device)
                div_loss_att2 += loss.diversity_loss(att2[k], args, device)
            all_losses += args.lambda_param*(div_loss_att1 + div_loss_att2)
        
        # final output
        output1_all = torch.zeros(output1[list(models.keys())[0]].data.shape)
        output1_all = output1_all.to(device)
        output2_all = torch.zeros(output2[list(models.keys())[0]].data.shape)
        output2_all = output2_all.to(device)
        for k in models.keys():
            output1_all += output1[k].data
            output2_all += output2[k].data

        # measure accuracy 
        prec, cor = util.accuracy(output1_all, output2_all)

        # record losses
        av_meters['ranking_losses'].update(ranking_loss.item(), input1.size(0)*len(models.keys()))
        if args.diversity_loss:
            av_meters['diversity_losses'].update(args.lambda_param*(div_loss_att1.item()+div_loss_att2.item()),
                                                 input1.size(0)*len(models.keys()))
        av_meters['losses'].update(all_losses.data.item(), input1.size(0))
        av_meters['acc'].update(prec, input1.size(0))
        av_meters['correct'].update(cor)

        # measure elapsed time
        av_meters['batch_time'].update(time.time() - begin)
        begin = time.time()

    console_log_test(av_meters)
    if log:
        tensorboard_log(av_meters, 'val', epoch)
    
    return av_meters['acc'].avg, True


def console_log_train_batch(av_meters, epoch, total_epoch, iter, 
                            loader_len, total_time, total):
    print(('Epoch:[{0}/{1}]Iter:[{2}/{3}]\t'
           'Time:{batch_time.val:.2f}s({4})\t\t'
           'Loss(Rank):{phase0_loss.val:.5f}\t'
           'Loss(Others):{phase1_loss.val:.5f}\t'
           'Accuracy:{acc.val:.3f}({correct.val}/{5})'.format(
               epoch, total_epoch, iter, loader_len, 
               util.sec2str(total_time), total, 
               batch_time=av_meters['batch_time'], loss=av_meters['losses'], 
               phase0_loss=av_meters['phase0_loss'], phase1_loss=av_meters['phase1_loss'],
               correct=av_meters['correct'], acc=av_meters['acc'])))

def console_log_train(av_meters, epoch, total_epoch, total_time):
    print(('[Epoch:({0}/{1}]) Results]\t'
           'Time:{batch_time.sum:.2f}s({2})\t\t'
           'Loss(Rank):{phase0_loss.avg:.5f}\t'
           'Loss(Others):{phase1_loss.avg:.5f}\t'
           'Accuracy:{acc.avg:.3f}({correct.sum}/{acc.count})'.format(
               epoch, total_epoch, util.sec2str(total_time), 
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

def tensorboard_log(av_meters, mode, epoch):
    writer.add_scalar(mode+'/total_loss', av_meters['losses'].avg, epoch)
    writer.add_scalar(mode+'/ranking_loss', av_meters['ranking_losses'].avg, epoch)
    writer.add_scalar(mode+'/diversity_loss', av_meters['diversity_losses'].avg, epoch)
    writer.add_scalar(mode+'/acc', av_meters['acc'].avg, epoch)

def tensorboard_log_with_uniform(av_meters, mode, epoch):
    tensorboard_log(av_meters, mode, epoch)
    writer.add_scalar(mode+'/disparity_loss', av_meters['disparity_losses'].avg, epoch)
    writer.add_scalar(mode+'/ranking_loss_uniform', av_meters['ranking_losses_uniform'].avg, epoch)
    writer.add_scalar(mode+'/acc_uniform', av_meters['acc_uniform'].avg, epoch)
    writer.add_scalar(mode+'/rank_aware_loss', av_meters['rank_aware_losses'].avg, epoch)
    writer.add_scalar(mode+'/phase0_loss', av_meters['phase0_loss'].avg, epoch)
    writer.add_scalar(mode+'/phase1_loss', av_meters['phase1_loss'].avg, epoch)



    
if __name__ == '__main__':
    main()


    
