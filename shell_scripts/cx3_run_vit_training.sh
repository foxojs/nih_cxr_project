#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -N nih_cxr

cd /rds/general/user/ojf24/home/ml_project/training
eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate tds_project 


python training_vit_scratch.py


