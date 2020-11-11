#!/bin/bash

# Train iPCRNet - all classes
python train_pcrnet.py \
    --datafolder modelnet40_ply_hdf5_2048 \
    --sampler none \
    --train-pcrnet \
    --epochs 400 \
    --exp_name iPCRNet1024_MN40

wait

# Train SampleNet - all classes
python train_pcrnet.py \
    --datafolder modelnet40_ply_hdf5_2048 \
    --sampler samplenet \
    --train-samplenet \
    --epochs 400 \
    --exp_name SAMPLENET64_on_iPCRNet1024_MN40 \
    --num-out-points 64 \
    --pretrained checkpoints/iPCRNet1024_MN40/models/best_model.t7

wait

# Train iPCRNet - car class
python train_pcrnet.py \
    --datafolder car_hdf5_2048 \
    --sampler none \
    --train-pcrnet \
    --epochs 400 \
    --exp_name iPCRNet1024_CAR

wait

# Train SampleNet - car class
python train_pcrnet.py \
    --datafolder car_hdf5_2048 \
    --sampler samplenet \
    --train-samplenet \
    --epochs 400 \
    --exp_name SAMPLENET64_on_iPCRNet1024_CAR \
    --num-out-points 64 \
    --pretrained checkpoints/iPCRNet1024_CAR/models/best_model.t7

wait
