#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=4:mem=50gb:ngpus=1
#PBS -N cx3_nih_gpu

cd /rds/general/user/ojf24/home/ml_project/vit_pretrained_fine_tune
eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate pytorch_env


python main.py


