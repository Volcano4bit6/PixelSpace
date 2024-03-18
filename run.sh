#!/bin/bash

find Data_Round1/Test_R1 -type f -exec readlink -f {} \; > Data_Round1/file_name_test.txt

chmod u+x get_pose.sh

bash get_pose.sh Data_Round1/file_name_test.txt

python LSTMAT/inference.py