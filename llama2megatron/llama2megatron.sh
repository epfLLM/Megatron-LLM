#!/bin/bash

# Navigate to the directory containing your Python script
cd /mlodata1/sfan/
LLAMA_CONFIG_PATH='/mlodata1/llms/llama/'

for SCALE in 7B 13B 30B 65B
do
    python llama2megatron.py --model_name $SCALE --llama_config_path $LLAMA_CONFIG_PATH;
done