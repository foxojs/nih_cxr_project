#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=20:mem=100gb
#PBS -N nih_cxr

cd /rds/general/user/ojf24/home/ml_project/vit_base_model
module load anaconda3/personal
source activate tds_project


python main.py

