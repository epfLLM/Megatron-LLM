#!/bin/bash

IMPL=cached
python ../preprocess_data.py \
       --input test_samples.json \
       --vocab vocab.txt \
       --dataset_impl ${IMPL} \
       --output_prefix test_samples_${IMPL} \
       --workers 1 \
       --log_interval 2
