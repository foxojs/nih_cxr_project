#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -N nih_cxr_gpu

cd /rds/general/user/ojf24/home/ml_project/vit_base_model
module load anaconda3/personal
source activate tds_project


python main.py
