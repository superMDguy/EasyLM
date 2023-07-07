#!/bin/bash

# Delete
for ((i=1; i<=4; i++))
do
    gcloud alpha compute tpus tpu-vm detach-disk cluster-node-$i --disk=clone --zone=europe-west4-a
done
gcloud compute disks delete clone --zone=europe-west4-a
exit 0

gcloud compute disks create projects/cloud-tpu-338714/zones/europe-west4-a/disks/clone \
    --description="cloned disk" \
    --source-disk=projects/cloud-tpu-338714/zones/europe-west4-a/disks/disk-1

for ((i=1; i<=4; i++))
do
    gcloud alpha compute tpus tpu-vm attach-disk cluster-node-$i \
        --zone=europe-west4-a \
        --disk=clone \
        --device-name=clone \
        --mode=read-only
    gcloud alpha compute tpus tpu-vm ssh cluster-node-$i --internal-ip --zone europe-west4-a \
     --command "sudo mkdir -p /mnt/disks/large && sudo mount -o ro,noload /dev/sdb /mnt/disks/large && sudo apt-get install -y libcairo2-dev pkg-config python3-dev && pip install -r /mnt/disks/large/EasyLM/requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
done
