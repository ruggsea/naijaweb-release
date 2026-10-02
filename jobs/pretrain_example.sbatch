#!/bin/bash
# Generic Slurm example for the pretraining runs (train/pretrain.py), one GPU per run.
#   sbatch -t <limit> pretrain_example.sbatch <passes> "<version> <seed>,<version> <seed>" "<next lane>" ...
# Each quoted argument is one lane of runs done in order, one GPU each (up to 4 lanes per 4-GPU node).
# A version is the name of a .bin file in train/bins/.
# A run whose final model already exists is skipped, so a preempted or failed job can simply be
# resubmitted.
#
# Replace the two #SBATCH lines below with your own account/partition/QoS and node's GPU count, and
# point at your own Slurm cluster's module/venv activation instead of the `source` line.
#SBATCH -A <YOUR_ACCOUNT> -p <YOUR_GPU_PARTITION> --qos <YOUR_QOS>
#SBATCH --gres=gpu:4 -J naijaweb_pretrain -o logs/job_%j.out

# source <your cluster's environment setup>; activate the repo's venv
: "${NAIJAWEB_DATA:?set NAIJAWEB_DATA}"
cd "$(dirname "$0")/.."   # repo root

PASSES=$1; shift
lane() {
  IFS=, read -ra runs <<< "$1"
  for run in "${runs[@]}"; do
    v=${run% *}; s=${run#* }
    [ -d models/pretrain_${v}_e${PASSES}_s$s ] && { echo "=== skip $v s$s (done)"; continue; }
    echo "=== $v s$s start $(date -u +%H:%MZ)"
    # the node is split into one-GPU tasks, so each run is its own job step with its own card; --mem
    # because a step otherwise asks for the whole node's memory and later steps wait on busy nodes
    srun -n1 -c22 --mem=160G --gres=gpu:1 --exact python train/pretrain.py --version "$v" --seed "$s" --passes "$PASSES" > logs/${v}_e${PASSES}_s$s.log 2>&1
    echo "=== $v s$s exit $? $(date -u +%H:%MZ)"
  done
}
for l in "$@"; do lane "$l" & done
wait
