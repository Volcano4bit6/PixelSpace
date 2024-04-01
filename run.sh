#!/bin/bash

tail -n +2 Data_Round1/input.csv | awk -F',' '{print $1}' > Data_Round1/input.txt

chmod u+x get_pose.sh

bash get_pose.sh Data_Round1/input.txt

cd LSTMAT

python inference.py