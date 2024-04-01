#!/bin/bash

# Check if the argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi
# Read each line from the file and echo it
while IFS= read -r line; do
    python mmpose/demo/inferencer_demo.py "$line" --pose3d human3d --pred-out-dir "$(dirname "$line")/pose_new" --vis-out-dir "$(dirname "$line")/pose_new"
    echo "$line"
done < "$1"