#!/bin/bash
# dir to do this on is first argument

demos=$(find $1 -maxdepth 1 -type d -printf '%f\n')
count=0

for d in $demos; do
    if [[ $count == 0 ]]; then
        count=1
        continue
    fi
    path=$(realpath "$1/$d")
    echo "Making vid for sts for $path"
    python -m contact_il.data.cam_to_vid $path 0
    echo "Making vid for wrist for $path"
    python -m contact_il.data.cam_to_vid $path 0 --data_key='wrist_rgb'

done
