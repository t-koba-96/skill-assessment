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
    parser.add_argument('--save_dir', type=str, default='../demo/videos', help='path to save frame')
    parser.add_argument('--frame_samples', type=int, default=400, help='保存するフレーム数')
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
            n_frames, frame_count = video_2_uniform_frames(video_dir = dir, video_name = name, save_dir = os.path.join(args.save_dir, task), 
                            frame_samples = args.frame_samples, frame_format = args.frame_format, frame_name_num = args.frame_name_num)
            if n_frames != args.frame_samples:
                print("Error : Frame num was {}/{}".format(n_frames, frame_count))
                return
            else:
                print('Saved {} ({}/{}) Total Frame : {}/{}'.format(name, i+1, vid_len, n_frames, frame_count))

        print('\n')
            

def video_2_uniform_frames(video_dir, video_name, save_dir, frame_samples = 400, frame_format = 'png', frame_name_num = 5):

    # Delete the entire directory tree if it exists.
    if os.path.exists(os.path.join(save_dir, video_name)):
        shutil.rmtree(os.path.join(save_dir, video_name)) 
    os.makedirs(os.path.join(save_dir, video_name))

    # Video 
    cap = cv2.VideoCapture(video_dir)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get sample frames list
    if frame_count >= frame_samples:
        sample_list = shrink_uniform_num(frame_count, frame_samples)
    else:
        sample_list = expand_uniform_num(frame_count, frame_samples)

    # Save frames
    i = 0
    count = 0
    while(cap.isOpened()):
        flag, frame = cap.read() 
        if flag == False:  
            break
        else:
            i += 1
            if i in sample_list:
                for sam in sample_list:
                    if i == sam:
                        # ５桁の数字で保存　例：　1 →　00001
                        count += 1
                        cv2.imwrite(os.path.join(save_dir, video_name, str(count).zfill(frame_name_num) + '.' + frame_format), frame) 
    cap.release()

    return count, frame_count


def shrink_uniform_num(count, sample):
    rate = count/sample
    sample_list = []
    j=1
    for j in range(1,sample+1):
        sample_list.append(int((j*rate)//1))
    return sample_list

def expand_uniform_num(count, sample):
    rate = int((sample/count)//1)
    rest = sample%count
    sample_list = []
    uni_list = shrink_uniform_num(count, rest)
    for j in range(1, count+1):
        if j in uni_list:
            for k in range(rate+1):
                sample_list.append(j)
        else:
            for k in range(rate):
                sample_list.append(j)
    return sample_list


if __name__ == '__main__':
    main()