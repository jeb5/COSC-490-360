#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
cd "$SCRIPT_DIR"

set -o allexport
source .env
set +o allexport

REMOTE_PATH="/home/nicje229/projects/cosc490_experiments/"
HOST="aoraki-login-stu.uod.otago.ac.nz"

# Use rsync to copy files, excluding .copy_ignore files
echo "Copying files to Aoraki..."
sshpass -p "$AORAKI_PASSWORD" rsync -ah --info=progress2 --delete --filter=":- .copy_ignore" . "$AORAKI_USERNAME@$HOST:$REMOTE_PATH/code/"

# Submit the job on the server
echo "Submitting job to Aoraki..."
sshpass -p "$AORAKI_PASSWORD" ssh $AORAKI_USERNAME@$HOST "bash -l -c \"cd $REMOTE_PATH/code && sbatch job.sh\""
