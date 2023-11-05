# sudo docker run --gpus all -it \
# 	--rm --shm-size=128gb \
# 	-v /pure-mlo-scratch:/pure-mlo-scratch \
# 	--network host \
# 	-v /home/zeming/model-parallel-trainer/:/main \
#         meditron \
#         /main/examples/pretrain_meditron.sh
 
sudo docker run --privileged \
        -v /pure-mlo-scratch/:/pure-mlo-scratch/ \
        --cap-add=IPC_LOCK \
        --device=/dev/infiniband/uverbs0 \
        --ipc=host --shm-size=128gb \
        --memory 480G --ulimit memlock=-1 \
        --ulimit stack=67108864 --rm \
        --gpus all -it \
        --network host \
        -v /home/zeming/Megatron-LLM/:/main \
        meditron \
        bash /main/meditron_run/pretrain_meditron.sh