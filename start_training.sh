#!/bin/bash

source activate /data/rauschecker1/brain_env

model_type='uk_biobank'
mods='T1|T2'
model_name_base='_registered_biobank_multimodal_dgx_fs-multi_regress_%06d'
rt_sv_dir='/data/rauschecker1/infantBrainAge/models'
ngpus=2
scale="1"
batch_size=16
learn_rate=1e-3
n_iter=5
exclude_abnorm=true
exclude_con=false
colors_to_inc='GREEN|YELLOW|ORANGE|RED'


python network_regress_model.py \
    --model_type=$model_type \
    --mods=$mods \
    --model_name_base=$model_name_base \
    --rt_sv_dir=$rt_sv_dir \
    --ngpus=$ngpus \
    --scale=$scale \
    --batch_size=$batch_size \
    --learn_rate=$learn_rate \
    --n_iter=$n_iter \
    --exclude_abnorm=$exclude_abnorm \
    --exclude_con=$exclude_con \
    --colors_to_inc=$colors_to_inc