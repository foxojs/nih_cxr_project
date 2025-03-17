#PBS -l walltime=18:00:00
#PBS -l select=1:ncpus=30:mem=120gb:ngpus=1
#PBS -N cx3_nih_gpu

cd /rds/general/user/ojf24/home/ml_project/vit_base_model
eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate tds_project 


python main.py


