source .venv/bin/activate
module load system/CUDA/12.2.0
export HF_HOME="$(pwd)/.hf"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True