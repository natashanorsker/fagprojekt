#!/bin/sh
#BSUB -J deep_ranking_gridsearch
#BSUB -o deep_ranking_gridsearch%J.out
#BSUB -e deep_ranking_gridsearch%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=8G]"
#BSUB -W 4:00
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load python3/3.8.4

# load CUDA (for GPU support)
module load cuda/11.1

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source ../test-env/bin/activate

python deep_ranking_gridsearch.py

