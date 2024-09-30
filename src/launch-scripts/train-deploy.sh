#!/bin/bash
#SBATCH --nodes=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=128gb
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --output=slurm-%j-%N.out
#SBATCH --signal=USR2@300

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

source ~/.bashrc
conda activate polyoculus

MAIN_HOST=`hostname -s`
export MASTER_ADDR=$MAIN_HOST
export MASTER_PORT=`python free-port.py $MASTER_ADDR`

export NCCL_IB_DISABLE=1

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

cd ../scripts
srun --nodes=$SLURM_JOB_NUM_NODES python train.py -c realestate-multiview_ldm.yaml --num_nodes $SLURM_JOB_NUM_NODES --num_devices $SLURM_NTASKS_PER_NODE

wait
