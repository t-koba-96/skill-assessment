# Skill Assessment for long range videos

## Data  

### Dataset

- [BEST Dataset](./data/BEST) - The Bristol Everyday Skill Tasks Dataset. Includes 5 tasks -> (apply_eyeliner, braid_hair, origami, scrambled_eggs, tie_tie) 

- [EPIC-Skills](./data/EPIC-Skills) - Epic skills Dataset. Includes 4 tasks -> (chopstick_using, dough_rolling, drawing, surgery) 

### Video Download

Only BEST Dataset available.  

```
cd data
python make_dataset/download_videos.py BEST/BEST.csv BEST/videos --trim 
```
 
--trim makes the video trimmed to the part which is only used as the dataset(No background sequenes).  

### Feature Download

Extracted i3d features for both BEST Dataset and EPIC-Skills .

```
cd data
bash make_dataset/download_features.sh 
```

###  Splits  

- [BEST Dataset](./data/BEST)
  - [data/BEST/splits/<task_name>/<train|test>_vid_list.txt] -> Train and test(valid) splits for each task.
  - [data/BEST/splits/<task_name>/<train|test>.txt] -> Video pairs for prediction. 

- [EPIC-Skills](./data/EPIC-Skills)
  - [data/EPIC-Skills/splits/<task_name>/<train|test>_split<split_num>.txt] -> Video pairs for each split is divided into files. 

### (Edited 2020/12/8)  
Because the dataset is based on youtube, some videos are missing after you run download_videos.py. To arrange the splits to your downloaded videos size, run the below code.  
Also dont forget to set arg[video_sets] to "videos" in the args file.

```
cd data
python make_dataset/make_new_splits.py
```


## Train

To train skill assessment models.

### Args  

#### Template Args

4 template args(config yaml files) are in [results](./results) directory.

- [origin.yaml](./results/origin/arg.yaml) ... model in the original paper
- [tcn.yaml](./results/tcn/arg.yaml)  ... temporal model template
- [new.yaml](./results/new/arg.yaml)  ... using new features
- [new_origin.yaml](./results/new_origin/arg.yaml)  ... new feature attached to origin model

#### Make New Args  

To make new features, run [./main/make_arg.py](./main/make_arg.py).  
For example, if you want to change m1 in [new.yaml](./results/new/arg.yaml) to 0.5, run 

```
python main/make_arg.py new --m1 0.5
```



### Train each task

For training each task independently use [./main/train.py](./main/train.py) as:

```
python main/train.py [arg_file] [task] [lap_num] [--cuda [gpu_num]]  
```  

### Train all task

For training all tasks together use [./main/train_all.sh](./main/train_all.sh) as:

```
bash main/train_all.sh [arg_file] [lap_num] [--cuda(-c) [gpu_num]]
```

Use Help option (--help) for more info.  
- Ablation Study  



### Results

Results of model and tensorboard log are saved in [results](./results) directory.

- Model weights and records
  - The best score is saved as (best_score_ ... .ckpt).
  - to change saved model frequency, change ckpt_freq in arg file

- Tensorboard logs
  - Tensorboard logs for train and eval are saved.

  (Private note : when starting tensorboard log on lab server)
  ```
  tensorboard --logdir results --port 8888 --bind_all
  ```


## Evaluate  

Eval results are used for demo.

### Eval each task

For evaluating each task independently use [./main/eval.py](./main/eval.py) as:

```
python main/eval.py [arg_file] [task] [lap_num] [--epoch [epoch_num]] [--cuda [gpu_num]]  
```  

### Train all task

For evaluating all tasks together use [./main/train_all.sh](./main/train_all.sh) as:

```
bash main/eval_all.sh [arg_file] [lap_num] [--epoch [epoch_num]] [--cuda(-c) [gpu_num]]
```

Use Help option (--help) for more info.   

## Demo  

To show demo, run [./main/demo.py](./main/demo.py) as  

```
python main/demo.py
```


## Working File  

- [test.ipynb](./test.ipynb)


## References  
'The Pros and Cons: Rank-aware Temporal Attention for Skill Determination in Long Videos'
