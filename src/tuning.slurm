#!/bin/bash
#SBATCH --nodelist=node5-3
#SBATCH --ntasks=1
#SBATCH --partition=GTX780
#SBATCH --mail-user=vladimir.shadymov@gmail.com 
#SBATCH --mail-type=END
#SBATCH --comment="STT_BNN"
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --job-name=c10_stt

module load cuda/10.0
srun python3 -u tuning_simulation.py --batch-size=2 --lr=1e-2 --test-batch-size=2 --epoch=100 --save-model --cuda-num=0 > ../log/mnist_bnn_tuning.log
