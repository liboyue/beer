#!/bin/bash
WORLD_SIZE=10

# nproc --all
ENVS="OMP_NUM_THREADS=1 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=$((9999 + `shuf -i 1-1000 -n 1`)) \
RANK=\${RANK} \
WORLD_SIZE=\${WORLD_SIZE} \
LOCAL_RANK=\${RANK} \
WORLD_LOCAL_SIZE=\${WORLD_SIZE} \
WORLD_NODE_RANK=0"

# Prepare parames
if [ "$1" == "-i" ]; then
    INTERACTIVE='-i'
    PARAMS="${PARAMS} ${@:2}"
else
    PARAMS="${PARAMS} ${@}"
fi

# Start processes
for (( RANK=$WORLD_SIZE-1; RANK>=0; RANK-- )) do
    if [[ "${RANK}" -eq 0 ]]; then
        eval "${ENVS} python ${INTERACTIVE} ${SCRIPT} ${PARAMS}"
    else
        eval "${ENVS} python ${SCRIPT} ${PARAMS}" &
    fi
done
