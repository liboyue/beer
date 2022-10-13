#!/bin/bash
SCRIPT='mnist.py'

TRAIN_PARAMS="--epochs=150 --batch_size=100 --val_interval=100 --disp_interval=50 --deterministic --num_workers=2"
TRAIN_PARAMS="--epochs=10 --batch_size=100 --val_interval=100 --disp_interval=50 --deterministic --num_workers=2"

OPTIM_PARAMS='--optimizer=CHOCO_SGD --lr=0.0001 --gamma=0.01 --compression_type=gsgd --compression_params=5'

DIST_PARAMS='--graph_type=er --graph_params=0.6 --backend=gloo'
eval "PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ./run.sh ${@} "
