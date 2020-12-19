import argparse
import glob
import os
import shutil
import cv2
import numpy as np
import torch
from torch import nn

'''
This code is only available for BEST dataset
'''


def get_arguments():
    parser = argparse.ArgumentParser(description='make_uniform_features')
    parser.add_argument('--dataset_dir', type=str, default='BEST/new_features_origin', help='path to video features')
    parser.add_argument('--save_dir', type=str, default='BEST/new_features', help='path to save features')
    parser.add_argument('--uniform_len', type=int, default=400, help='uniform samples')
    parser.add_argument('--feature_format', type=str, default='npy', help='feature format [npy, npz]')
    return parser.parse_args()

def main():
    args = get_arguments()
    task_list = ["origami", "scrambled_eggs", "tie_tie"]

    for task in task_list:
        feat_paths = glob.glob(os.path.join(args.dataset_dir, task, '*'))
        # extract only feature_file_name from feature_path
        feat_names = [os.path.splitext(os.path.basename(path))[0] for path in feat_paths]
        vid_len = len(feat_names)

        # Delete the entire directory tree if it exists.
        if os.path.exists(os.path.join(args.save_dir, task)):
            shutil.rmtree(os.path.join(args.save_dir, task)) 
        os.makedirs(os.path.join(args.save_dir, task))

        print('[Task {} start ({})]'.format(task, vid_len))

        for i, (dir, name) in enumerate(zip(feat_paths, feat_names)):
            uniform_features(video_dir = dir, video_name = name, save_dir = os.path.join(args.save_dir, task), 
                             uniform_len = args.uniform_len, feature_format = args.feature_format)
            print('Saved {}.{} ({}/{})\n'.format(name, args.feature_format, i+1, vid_len))

        print('\n')
            


def uniform_features(video_dir, video_name, save_dir, uniform_len, feature_format = 'npy'):

    # load feature
    feature = np.load(os.path.join(video_dir)).astype(np.float32)
    tensor = t = torch.tensor(feature)
    uniform_pooling = nn.AdaptiveAvgPool1d(uniform_len)

    new_tensor = uniform_pooling(tensor.view(tensor.size(0),tensor.size(1), -1).permute(2,1,0)).permute(2,1,0).view(-1, tensor.size(1), tensor.size(2), tensor.size(3))
    print("<{}> : {} → {}".format(video_name, tensor.size(), new_tensor.size()))
    np.save(os.path.join(save_dir, video_name+"."+feature_format), new_tensor.cpu())


if __name__ == '__main__':
    main()