#!/bin/sh
#BSUB -J online_triplet
#BSUB -o online_triplet%J.out
#BSUB -e online_triplet%J.err
#BSUB -n 16
#BSUB -R "rusage[mem=12G]"
#BSUB -W 1:00
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
