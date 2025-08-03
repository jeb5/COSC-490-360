#!/bin/bash

REMOTE_PATH="/home/nicje229/projects/cosc490_experiments/"
USERNAME="nicje229"
HOST="aoraki-login-stu.uod.otago.ac.nz"

read -s -p "Enter your password for Aoraki: " PASSWORD
echo
password_correct=0
while [ $password_correct -eq 0 ]; do
  sshpass -p "$PASSWORD" ssh $USERNAME@$HOST "echo 'Password is correct'" && password_correct=1 || {
    read -s -p "Enter your password for Aoraki: " PASSWORD
    echo
  }
done

# Use rsync to copy files, excluding .copy_ignore files
echo "Copying files to Aoraki..."
sshpass -p "$PASSWORD" rsync -ah --info=progress2 --delete --filter=":- .copy_ignore" . "$USERNAME@$HOST:$REMOTE_PATH/code/"

# Submit the job on the server
echo "Submitting job to Aoraki..."
sshpass -p "$PASSWORD" ssh $USERNAME@$HOST "bash -l -c \"cd $REMOTE_PATH/code && sbatch job.sh\""
