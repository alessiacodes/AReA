#!/bin/bash
#SBATCH --job-name=area-infer4k
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

# Inference-only AReA (FLUX.1-Krea-dev) run with latency logging, same settings as generate.py --model flux1.
# Slurm refuses jobs submitted from /home, so submit from ~/scratch/area_runs (the Slurm log lands there):
#   cd ~/scratch/area_runs && PROMPT="..." sbatch ~/AReA/infer_4k_timed.sbatch
# Optional overrides: RUNS=1 SEED=... REPO=... EXTRA_ARGS=--no_hooks PROMPT="..." sbatch [--gres=gpu:h100:1] ~/AReA/infer_4k_timed.sbatch
# zrun in the repo is a symlink to ~/scratch/area_runs, so images and logs go to scratch.

cd "${REPO:-$HOME/AReA}" || exit 1

module load python/3.11 cuda/12.9
source ~/scratch/envs/area/bin/activate   # torch 2.11.0+cu128; other packages from requirements.txt resolved as of 2026-10-14

export HF_HOME=~/scratch/hf_cache   # FLUX.1-Krea-dev is cached here; the HF token is in $HF_HOME/token
# Not setting HF_HUB_OFFLINE=1 (it broke model loading for UltraImageGen and JoLT).
export TRITON_CACHE_DIR=~/scratch/.triton_cache
export CUDA_CACHE_PATH=~/scratch/.nv/ComputeCache
export XDG_CACHE_HOME=~/scratch/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1
: "${PROMPT:?set PROMPT}"
RUNS=${RUNS:-2}
SEED=${SEED:-42}

RUN_DIR=zrun/infer4k-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is) run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Sample GPU memory and utilization once per second for the whole job.
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > "$RUN_DIR/gpu_usage.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

python infer_timed.py --prompt "$PROMPT" --seed "$SEED" --out_dir "$RUN_DIR" \
    --runs "$RUNS" --time_budget_s 10200 ${EXTRA_ARGS}
STATUS=$?

kill $SMI_PID 2>/dev/null
awk -F', ' '{ if ($2 + 0 > m) m = $2 + 0; u += $3; n++ }
     END { if (n) printf "nvidia-smi: peak memory.used %d MiB, mean GPU util %.0f%% over %d s sampled\n", m, u / n, n }' \
    "$RUN_DIR/gpu_usage.csv" | tee -a "$RUN_DIR/summary.txt"
echo "end=$(date -Is) exit=$STATUS"
exit $STATUS
