#PBS -l walltime=18:00:00
#PBS -l select=1:ncpus=30:mem=130gb:ngpus=1:gpu_type=RTX6000
#PBS -N nih_cxr_gpu

cd /rds/general/user/ojf24/home/ml_project/vit_pretrained_fine_tune
module load anaconda3/personal
source activate pytorch_env


python main.py
