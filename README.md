# AI4LIFE 2024
# PIXEL SPACE TEAM
## Contributors: T.Van Hong Son, L.Viet Hung
## Environment set up:
* `conda create --name openmmlab python=3.8 -y`
* `conda activate openmmlab`
* `conda install pytorch torchvision -c pytorch`
* `pip install -U openmim`
* `mim install mmengine`
* `mim install "mmcv>=2.0.1"`
* `mim install "mmdet>=3.1.0"`
* `cd mmpose`
* `pip install -r requirements.txt`
* `pip install -v -e .`
* `mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .`
* `cd ..`
* `cd LSTMAT`
* `pip install -r requirements.txt`
* `wget https://huggingface.co/allasobi110/LSTMAT_AI4L/resolve/main/LSTM_Attention_128HUs_f1.h5?download=true`
## Folder set up:
*     Root_Folder:
    	|____ LSTMAT
     	|____ mmpose
        |____ Data_Round1/Data_R1
                           |_____ barbell biceps curl
                           |		|____tên video_1.mp4
                           |		|____……………
                           |		|____tên video_n.mp4
                           |_____ bench press
                           |		|____tên video_1.mp4
                           |		|____……………
                           |		|____tên video_n.mp4
                           |.......................................
                           |_____ tricep Pushdown
                                      |____tên video_1.mp4
                                      |____……………
                                      |____tên video_n.mp4
## Inference:
* `chmod u+x run.sh`
* `./run.sh`
* The result is an image in folder LSTMAT named PIXCEL_SPACE_CORR_F1_RESULT-AAAAAAAAAAA.png
