python train.py data/BEST/splits/apply_eyeliner/train.txt data/BEST/splits/apply_eyeliner/test.txt data/BEST/features/apply_eyeliner -e --transform --attention --diversity_loss --disparity_loss --rank_aware_loss

python train.py data/BEST/splits/braid_hair/train.txt data/BEST/splits/braid_hair/test.txt data/BEST/features/braid_hair -e --transform --attention --diversity_loss --disparity_loss --rank_aware_loss

python train.py data/BEST/splits/scrambled_eggs/train.txt data/BEST/splits/scrambled_eggs/test.txt data/BEST/features/scrambled_eggs -e --transform --attention --diversity_loss --disparity_loss --rank_aware_loss

python train.py data/BEST/splits/tie_tie/train.txt data/BEST/splits/tie_tie/test.txt data/BEST/features/tie_tie -e --transform --attention --diversity_loss --disparity_loss --rank_aware_loss