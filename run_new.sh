#!/bin/bash
# Check if the argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi
chmod u+x get_pose.sh

bash get_pose.sh 

python LSTMAT/inference.py