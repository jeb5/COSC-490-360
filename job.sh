#!/bin/bash
#SBATCH --account=nicje229
#SBATCH --job-name=COSC490_360_Experiments
#SBATCH --partition=aoraki_gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --output=job_%j.log

set -euo pipefail

module load cuda/11.8
module load python/3.10.8

PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"
ENV_DIR="$PROJECT_ROOT/../360_env"

mkdir -p "$ENV_DIR"
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

bash "$PROJECT_ROOT/setup.sh"

# ------ Ensure ffmpeg is available ------
FFMPEG_DIR="$ENV_DIR/ffmpeg"
export PATH="$FFMPEG_DIR:$PATH"

# Check if ffmpeg is in the PATH
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found, downloading static build..."
		mkdir -p "$FFMPEG_DIR"
		wget -qO- https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar -xJ --strip-components=1 -C "$FFMPEG_DIR"
		echo "ffmpeg downloaded"
fi
echo "ffmpeg version: $(ffmpeg -version | head -n 1)"
# ---------------------------------------

echo "Running COSC490 experiments..."
python -u "$PROJECT_ROOT/scripts/run_experiments.py"

deactivate

echo "Job finished."