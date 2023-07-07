#!/bin/bash

for ((i=1; i<=4; i++))
do
    echo "Running on cluster-node-$i"
    gcloud alpha compute tpus tpu-vm ssh cluster-node-$i --zone=europe-west4-a --internal-ip \
         --command="cd /mnt/disks/large/EasyLM; PROCESS_ID=$i  ./scripts/serve_llama_3b.sh"
    
    # gcloud alpha compute tpus tpu-vm ssh cluster-node-$i --zone=europe-west4-a \
    #      --command="sudo apt-get install -y libcairo2-dev pkg-config python3-dev && pip install -r /mnt/disks/large/EasyLM/requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
done