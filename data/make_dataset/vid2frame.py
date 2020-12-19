import argparse
import glob
import os
import os
import shutil
import cv2

'''
This code is only available for BEST dataset
'''


def get_arguments():
    parser = argparse.ArgumentParser(description='make_video_2_frames')
    parser.add_argument('--dataset_dir', type=str, default='BEST/videos', help='path to video dataset')
    parser.add_argument('--save_dir', type=str, default='BEST/vid_frames', help='path to save frame')
    parser.add_argument('--frame_name_num', type=int, default=5, help='保存するフレームの名前(数字の桁数)')
    parser.add_argument('--frame_format', type=str, default='png', help='frame format [png, jpg]')
    return parser.parse_args()

def main():
    args = get_arguments()
    task_list = ["apply_eyeliner", "braid_hair", "origami", "scrambled_eggs", "tie_tie"]

    for task in task_list:
        vid_paths = glob.glob(os.path.join(args.dataset_dir, task, '*'))
        # extract only video_file_name from file_path
        vid_name = [os.path.splitext(os.path.basename(path))[0] for path in vid_paths]
        vid_len = len(vid_name)

        print('[Task {} start ({})]'.format(task, vid_len))

        for i, (dir, name) in enumerate(zip(vid_paths, vid_name)):
            n_frames = video_2_frames(video_dir = dir, video_name = name, save_dir = os.path.join(args.save_dir, task), 
                            frame_format = args.frame_format, frame_name_num = args.frame_name_num)
            print('Saved {}.mp4 ({}/{}) Total Frame : {}'.format(os.path.join(name), i+1, vid_len, n_frames))

        print('\n')
            


def video_2_frames(video_dir, video_name, save_dir, frame_format = 'png', frame_name_num = 5):

    # Delete the entire directory tree if it exists.
    if os.path.exists(os.path.join(save_dir, video_name)):
        shutil.rmtree(os.path.join(save_dir, video_name)) 

    os.makedirs(os.path.join(save_dir, video_name))

    # Video to frames
    i = 0
    cap = cv2.VideoCapture(video_dir)
    while(cap.isOpened()):
        flag, frame = cap.read()  
        if flag == False:  
            break
        # ５桁の数字で保存　例：　1 →　00001
        cv2.imwrite(os.path.join(save_dir, video_name, str(i).zfill(frame_name_num) + '.' + frame_format), frame) 
        i += 1
    cap.release()

    return i


if __name__ == '__main__':
    main()