#!/bin/sh
#BSUB -J online_triplet_gpu
#BSUB -o online_triplet_gpu%J.out
#BSUB -e online_triplet_gpu%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=8G]"
#BSUB -W 1:00
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.5.3-python-3.8.4

# load CUDA (for GPU support)
module load cuda/11.1

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source test-env/bin/activate

python deepRanking/train_online_model.py

echo $CPUTYPEV
