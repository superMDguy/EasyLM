#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Usage: $0 <number>"
  exit 1
fi

n=$1

if ! [[ $n =~ ^[0-9]+$ ]]; then
  echo "Error: Invalid number '$n'"
  exit 1
fi

PREFIX="cluster-node"

# Function to create a single node
create_node() {(
  set -e
  local i=$1
  echo "Creating Node $i"
  # --data-disk source=projects/cloud-tpu-338714/zones/europe-west4-a/disks/disk-1,mode=read-only \
  gcloud compute tpus tpu-vm create "$PREFIX-$i" \
    --zone=us-central1-f \
    --accelerator-type=v2-8 \
    --version=tpu-ubuntu2204-base \
    --preemptible
  # gcloud alpha compute tpus tpu-vm scp --zone=us-central1-f /home/supermdguy/EasyLM/distributed_test/main.py $PREFIX-$i:~/main.py
  # gcloud alpha compute tpus tpu-vm ssh --zone=us-central1-f $PREFIX-$i --command="pip3 install -r requirements.txt && python main.py --server_addr=\"10.164.0.19:1456\" --num_hosts=2 --host_idx=$i"
  # gcloud alpha compute tpus tpu-vm ssh --zone=us-central1-f $PREFIX-$i --command="pip3 install ray && ~/.local/bin/ray start --address=10.164.0.19:6379"
  gcloud alpha compute tpus tpu-vm ssh --zone=us-central1-f $PREFIX-$i --command="pip3 install ray && python -c \"import ray; ray.init('10.164.0.19:6379')\""
)}

# Set the chunk size
chunk_size=8

# Calculate the number of chunks
num_chunks=$((($n + $chunk_size - 1) / $chunk_size))

# Iterate over chunks
for ((chunk = 0; chunk < $num_chunks; chunk++)); do
  start=$((chunk * chunk_size + 1))
  end=$((start + chunk_size - 1))
  end=$((end > n ? n : end))
  
  # Create nodes in parallel for the current chunk
  for ((i = start; i <= end; i++)); do
    create_node "$i" &
  done
  
  # Wait for all background processes to finish
  wait
done
