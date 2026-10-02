# jobs/

A generic Slurm sbatch example for the GPU-heavy step: pretraining. Fill in your own
`#SBATCH -A/-p/--qos` and GPU count; it deliberately has no cluster-specific account, partition,
or hostname.

- `pretrain_example.sbatch` — the pretraining runs (`train/pretrain.py`), one GPU per run, several
  runs per node.

Resumable: a run whose output already exists is skipped, so a preempted or failed job can be
resubmitted as-is.
