sudo docker run --gpus 0 -it --rm --shm-size=2gb -v /scratch/pagliard/:/scratch --network host -v /home/pagliard/:/mpt epfllm -- /bin/bash -c 'bash /scratch/model-parallel-trainer/tokenize-utils/entrypoint.sh'
