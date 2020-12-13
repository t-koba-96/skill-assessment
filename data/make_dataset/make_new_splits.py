import argparse

import os 
import glob
import pandas as pd


def get_arguments():

    parser = argparse.ArgumentParser(description='make_new_splits')
    parser.add_argument('--dataset', type=str, default= 'BEST', help='dataset_name')
    parser.add_argument('--save', type=str, default= 'new_splits', help='save_folder_name')
    return parser.parse_args()


def main():
    # settings
    args = get_arguments()
    task_list = ["apply_eyeliner", "braid_hair", "origami", "scrambled_eggs", "tie_tie"]
    tt_list = ["train", "test"] 
    os.makedirs(os.path.join(args.dataset, args.save), exist_ok = True)

    # make vid list
    all_vid_list, install_vid_list, lack_vid_list = make_vid_list(args, task_list)

    # make txt file
    make_txt_file(args, task_list, tt_list, all_vid_list, install_vid_list, lack_vid_list)



def make_vid_list(args, task_list):
    all_vid_df = pd.read_csv(os.path.join(args.dataset, "BEST.csv"), sep=',', encoding='utf-8', index_col=False, header=None)
    all_vid_list = {}
    install_vid_list = {}
    lack_vid_list = {}
    for task in task_list:
        all_vid_list[task] = []
        install_vid_list[task] = []
        lack_vid_list[task] = []
        
        # all vid list
        for task_name, vid_name in zip(list(all_vid_df[0]), list(all_vid_df[1])):
            if task_name == task:
                all_vid_list[task].append(vid_name)
                
        # installed vid list
        installed_list = glob.glob(os.path.join(args.dataset,"videos", task, "*"))
        for file_path in installed_list:
            install_vid_list[task].append(os.path.splitext(os.path.basename(file_path))[0])
        print("{} : {}videos".format(task, len(install_vid_list[task])))
        
        #lacked vid list
        for file_name in all_vid_list[task]:
            if file_name not in install_vid_list[task]:
                lack_vid_list[task].append(file_name)

    return all_vid_list, install_vid_list, lack_vid_list



def make_txt_file(args, task_list, tt_list, all_vid_list, install_vid_list, lack_vid_list):

    # for each task
    for task in task_list:
        split_path = os.path.join(args.dataset, "splits", task) 
        new_split_path = os.path.join(args.dataset, args.save, task)
        os.makedirs(new_split_path, exist_ok = True)
        
        # for test, train each
        for tt in tt_list: 
            all_vid_list[task+"_"+tt] = []
            install_vid_list[task+"_"+tt] = []
            lack_vid_list[task+"_"+tt] = []
            all_vid_list[task+"_"+tt+"_pairs"] = []
            install_vid_list[task+"_"+tt+"_pairs"] = []
            lack_vid_list[task+"_"+tt+"_pairs"] = []
            
            # ===== first, edit vid_list.txt ====
            # save file
            new_txt_path = os.path.join(new_split_path, tt + '_vid_list.txt')
            w = open(new_txt_path, 'w')
            # read file
            not_found = []
            txt_path = os.path.join(split_path, tt + '_vid_list.txt')
            with open(os.path.join(txt_path), encoding="cp932") as f:
                for line in f.readlines():
                    all_vid_list[task+"_"+tt].append(line.rstrip('\n'))
                    if line.rstrip('\n') not in lack_vid_list[task]:
                        install_vid_list[task+"_"+tt].append(line.rstrip('\n'))
                        w.write(line) 
                    else:
                        lack_vid_list[task+"_"+tt].append(line.rstrip('\n'))
                        not_found.append(line.rstrip('\n'))
            print(not_found)
            w.close()
            
            # ===== second, edit .txt ====    
            # save file
            new_txt_path = os.path.join(new_split_path, tt + '.txt')
            w = open(new_txt_path, 'w')
            # read file
            not_found = []
            txt_path = os.path.join(split_path, tt + '.txt')
            with open(os.path.join(txt_path), encoding="cp932") as f:
                for line in f.readlines():
                    all_vid_list[task+"_"+tt+"_pairs"].append(line.rstrip('\n'))
                    missing = False
                    for l in line.rstrip('\n').split():
                        if l in lack_vid_list[task]:
                            missing = True
                    if not missing:
                        install_vid_list[task+"_"+tt+"_pairs"].append(line.rstrip('\n'))
                        w.write(line)
                    else:
                        lack_vid_list[task+"_"+tt+"_pairs"].append(line.rstrip('\n'))
                        not_found.append(line.rstrip('\n')) 
            w.close()

        # ==== last, make dataset.txt ====
        # save file
        dataset_path = os.path.join(new_split_path, 'dataset.txt')
        w = open(dataset_path, 'w')
        # write datas
        w.write('<{} Dataset>\n'.format(task))
        w.write('\n')
        w.write('[Train vid num]\n'.format(task))
        w.write('Origin : {} , Downloaded : {}\n'.format(len(all_vid_list[task+"_train"]), len(install_vid_list[task+"_train"])))
        w.write('[Test vid num]\n'.format(task))
        w.write('Origin : {} , Downloaded : {}\n'.format(len(all_vid_list[task+"_test"]), len(install_vid_list[task+"_test"])))
        w.write('\n')
        w.write('[Train vid_sets num]\n'.format(task))
        w.write('Origin : {} , Downloaded : {}\n'.format(len(all_vid_list[task+"_train_pairs"]), len(install_vid_list[task+"_train_pairs"])))
        w.write('[Test vid_sets num]\n'.format(task))
        w.write('Origin : {} , Downloaded : {}\n'.format(len(all_vid_list[task+"_test_pairs"]), len(install_vid_list[task+"_test_pairs"])))
        w.write('\n')
        w.write('[Train missing vids]\n'.format(task))
        w.write('{}\n'.format(lack_vid_list[task+"_train"]))
        w.write('[Test missing vids]\n'.format(task))
        w.write('{}\n'.format(lack_vid_list[task+"_test"]))
        w.close()



if __name__ == '__main__':
    main()


    
