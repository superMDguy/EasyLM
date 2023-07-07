#!/bin/bash

while true
do
    for ((i=1; i<=4; i++))
    do
        echo "Creating TPU VM cluster-node-$i"
        gcloud compute tpus tpu-vm create "cluster-node-$i" \
            --zone=europe-west4-a \
            --accelerator-type=v3-8 \
            --version=tpu-ubuntu2204-base
    done
    sleep 1800  # Sleep for 30 minutes
done