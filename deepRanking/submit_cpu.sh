#!/bin/sh
#BSUB -J mAP_subcategories
#BSUB -o mAP_subcategories%J.out
#BSUB -e mAP_subcategories%J.err
#BSUB -n 9
#BSUB -R "rusage[mem=8G]"
#BSUB -W 4:00
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load python3/3.8.4

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source ../test-env/bin/activate

python mAP.py
