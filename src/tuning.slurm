#!/bin/bash
#SBATCH --nodelist=node5-3
#SBATCH --ntasks=1
#SBATCH --partition=GTX780
#SBATCH --mail-user=shadymov.va@phystech.edu
#SBATCH --mail-type=END
#SBATCH --comment="STT_BNN"
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --job-name=bnn_sq

module load cuda/10.0
srun python3 -u tuning_simulation.py --batch-size=1 --lr=1e-2 --test-batch-size=10 --epoch=30 --log-interval=10000 --model-size=200 --save-model --cuda-num=0 > ../log/mnist_bnn_tuning_sq_200.log
