import os
import cv2
import time
import torch
import shutil
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch import nn



# ====== Runner class ======

class Eval_Runner():

    def __init__(self, opts, args, device, paths, model, dataloader):

        self.opts = opts
        self.args = args
        self.device = device
        self.dataloader = dataloader
        self.model = model
        self.att_branches = model.keys()
        self.resultdir = paths["resultdir"]
        self.demodir = paths["demodir"]

        self.checkpoint_dict = None



    '''
    Three type of records 
    1 : Score list -> csv
    2 : Temporal attention -> csv
    3 : Spatial attention -> png 
    '''
    def evaluate(self, epoch=10, is_best=False):
        # start epoch
        begin = time.time()

        # load best epoch model
        if is_best:
            print('[Evaluate : Using Best Epoch]')
            csv_name = "best_epoch"
            epoch = self.load_ckpt(is_best=True)
        # load epoch model
        else:
            print('[Evaluate : Using Epoch[{}/{}]'.format(epoch, self.args.epochs))
            csv_name =  str(epoch).zfill(4) + "_epoch"
            epoch = self.load_ckpt(epoch=epoch)

        # model evaluate mode
        for k in self.att_branches:
            self.model[k].eval()

        # csv record list
        score_list = {"pos" : [], "neg" : [], "pos_score" : [], "neg_score" : []}
        vid_name_list = {}
        te_att_list = {}
        for k in self.att_branches:
            vid_name_list[k] = []
            te_att_list[k] = None

        # run
        with torch.no_grad():
            # progress bar 
            with tqdm(total=len(self.dataloader.dataset)) as pbar:
                pbar.set_description(f"Epoch[{epoch}](Evaluate)")

                # ====== iter ====== 
                for i, (vid_pos, vid_neg, vid_list) in enumerate(self.dataloader):
                    batch_size = vid_pos.size(0)
                    for pos_neg, vid in zip(["pos", "neg"], [vid_pos, vid_neg]):
                        vid_gpu = vid.to(self.device)
                        score_list[pos_neg].extend(vid_list[pos_neg]) 

                        # calc score, attention
                        score, te_att, sp_att = {}, {}, {}
                        final_score = torch.zeros(batch_size).to(self.device)
                        for k in self.att_branches:
                            score[k], te_att[k], sp_att[k] = self.model[k](vid_gpu)
                            score[k] = score[k].mean(dim=1)
                            final_score += score[k].data
                            te_att[k] = te_att[k].mean(dim=2)
                            for index, vid_name in enumerate(vid_list[pos_neg]):
                                if vid_name not in vid_name_list[k]:
                                    pbar.set_postfix({"Video":"{}".format(vid_name)})
                                    te_att_list[k] = self.make_att_result(index, vid_name, k, sp_att[k], te_att[k], te_att_list[k])
                                    vid_name_list[k].append(vid_name)
                        score_list[pos_neg+"_score"].extend(final_score.cpu().numpy())

                    # progress bar
                    pbar.update(batch_size)

        correct_list = [score_list["pos_score"][i]>score_list["neg_score"][i] for i in range(len(score_list["pos_score"]))]

        # dataframe
        eval_df = pd.DataFrame({'vid_pos' : score_list["pos"], 
                                'vid_pos_score' : score_list["pos_score"], 
                                'vid_neg' : score_list["neg"], 
                                'vid_neg_score' : score_list["neg_score"], 
                                'correct' : correct_list
                            })
        eval_df.index = np.arange(1,len(score_list["pos_score"])+1)
        eval_df.to_csv(os.path.join(self.demodir, csv_name + "_score.csv"))
        print('Saving csv file ... {}'.format(csv_name + "_score.csv"))

        for k in te_att_list:
            att_df = pd.DataFrame(te_att_list[k],
                                  index = vid_name_list[k]
            )
            att_df.columns = np.arange(1,len(te_att_list[k][0])+1)
            att_df.to_csv(os.path.join(self.demodir, csv_name + "_" + k + ".csv"))
            print('Saving csv file ... {}'.format(csv_name + "_" + k + ".csv"))



    def load_ckpt(self, epoch=10, is_best=False):

        if is_best:
            weight_path = glob.glob(os.path.join(self.resultdir, 'best_score*'))[0]
        else:
            weight_path = glob.glob(os.path.join(self.resultdir, 'epoch_' + str(epoch).zfill(4) + '*'))[0]
        print('Loading checkpoint file ... {}'.format(weight_path))
        for k in self.att_branches:
            self.model[k].load_state_dict(torch.load(weight_path)['state_dict_' + k])
        print('Loaded model ... epoch : {:04d}  prec_score : {:.4f}'.format(torch.load(weight_path)["epoch"], 
                                                                            torch.load(weight_path)["prec_score"]))
        epoch = torch.load(weight_path)["epoch"]

        return epoch



    def make_att_result(self, index, vid_name, branch, sp_att, te_att, te_att_list):

        # spatial attention
        if self.args.spatial_attention:
            frame_paths = glob.glob(os.path.join(self.opts.demo_dir, "videos", self.opts.task, vid_name, '*'))
            save_path = os.path.join(self.demodir, 'video_'+branch, vid_name)
            if os.path.exists(save_path):
                shutil.rmtree(save_path) 
            os.makedirs(save_path)
            for i, path in enumerate(frame_paths):
                blend_img = make_heatmap(sp_att[index,i,0,:,:], path)
                cv2.imwrite(os.path.join(save_path, str(i+1).zfill(5)+'.png'), blend_img)
                
        # temporal attention
        pooling = nn.AdaptiveAvgPool1d(400)
        att = te_att[index:index+1]
        uniform_att = pooling(att.permute(0,2,1)).permute(0,2,1)
        if te_att_list is None:
            te_att_list = np.squeeze(uniform_att.cpu().detach().numpy(), 2)
        else:
            te_att_list = np.concatenate((te_att_list, 
                                    np.squeeze(uniform_att.cpu().detach().numpy(), 2)), axis=0)

        return te_att_list




def normalize_heatmap(x):
    # choose min (0 or smallest scalar)
    min = x.min()
    max = x.max()
    result = 1 - (x-min)/(max-min)
    
    return result

def make_heatmap(att_map, img_path, alpha = 0.7):
    # Image
    image = Image.open(img_path)
    img = np.asarray(image)
    # Heatmap
    heatmap = att_map.cpu().detach().numpy()
    heatmap = normalize_heatmap(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # Blend
    alpha = 0.7
    blend = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    return blend