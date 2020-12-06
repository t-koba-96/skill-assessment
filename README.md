# Skill Assessment for long range videos

## Data  

### Dataset

- [BEST Dataset](./data/BEST) - The Bristol Everyday Skill Tasks Dataset. Includes 5 tasks -> (apply_eyeliner, braid_hair, origami, scrambled_eggs, tie_tie) 

- [EPIC-Skills](./data/EPIC-Skills) - Epic skills Dataset. Includes 4 tasks -> (chopstick_using, dough_rolling, drawing, surgery) 

### Video Download

Only BEST Dataset available. 

```
cd data
python download/download_videos.py BEST/BEST.csv <download_dir> --trim 
```
 
--trim makes the video trimmed to the part which is only used as the dataset(No background sequenes).

### Feature Download

Extracted i3d features for both BEST Dataset and EPIC-Skills .

```
cd data
bash download/download_features.sh 
```

###  Splits  

- [BEST Dataset](./data/BEST)
  - [data/BEST/splits/<task_name>/<train|test>_vid_list.txt] -> Train and test(valid) splits for each task.
  - [data/BEST/splits/<task_name>/<train|test>.txt] -> Video pairs for prediction. 

- [EPIC-Skills](./data/EPIC-Skills)
  - [data/EPIC-Skills/splits/<task_name>/<train|test>_split<split_num>.txt] -> Video pairs for each split is divided into files. 


## Train

### Args

- Args for training are writen in yaml file. Default args are in [origin.yaml](./args/origin.yaml).  
- Make a new yaml file in the args directory to train on your on settings.  
- [arg_file] for training will be the name before .yaml.   
(For example, [arg_file] for origin.yaml will be origin)

### Train each task

For training each task independently use [train.py](train.py) as:

```
python train.py [arg_file] [dataset] [task] [lap_count] [--split [split for EPIC-Skills]] [--cuda [gpu_num]]  
```  

### Train all task

For training all tasks together use [run.sh](run.sh) as:

```
bash run.sh [arg_file] [dataset] [lap_count] [--split(-s) [split for EPIC-Skills]] [--cuda(-c) [gpu_num]]
```

Use Help option (--help) for more info.  

### Ablation Study  

To do . [ablationrun.sh](ablationrun.sh)

### Checkpoints

- Model weights  
  - For default, trained model weights are saved to [./ckpt/models](./ckpt/models)
  - Change the save directory if you want by changing the ckpt_path in [arg_file](./args/origin.yaml)
  - The best score is saved as (best_score_ ... .ckpt).
  - Also the best score results are saved as csv files.  

- Tensorboard logs
  - For default, tensorboard logs are saved to [./ckpt/logs](./ckpt/logs)
  - Change the log directory if you want by changing the writer_path in [arg_file](./args/origin.yaml)

  (Private note : when starting tensorboard log on lab server)
  ```
  tensorboard --logdir ckpt/logs --port 8888 --bind_all
  ```


## Evaluate  

To do .  


## Working File  

- [ex.ipynb](./ex.ipynb)


## References  
'The Pros and Cons: Rank-aware Temporal Attention for Skill Determination in Long Videos'
