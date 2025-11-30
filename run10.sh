#!/bin/bash

for i in {1..10}
do
    echo "Starting run $i"
    python experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --num_trials_per_task 10 \
        --use_constraints True \
        --center_crop True
done

