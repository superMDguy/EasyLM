python -m EasyLM.models.llama.llama_serve \
    --load_llama_config='65b' \
    --load_checkpoint="params::/mnt/disks/large/llama-65b.stream" \
    --tokenizer.vocab_file='/mnt/disks/large/tokenizer.model' \
    --dtype='bf16' \
    --input_length=256 \
    --seq_length=512 \
    --top_k=0 \
    --top_p=0.73 \
    --lm_server.batch_size=1 \
    --lm_server.port=35009 \
    --lm_server.pre_compile='all' 
#    --initialize_jax_distributed \
#   --jax_distributed.initialize_jax_distributed \
#   --jax_distributed.num_processes=5 \
#   --jax_distributed.coordinator_address='10.164.0.19' \
#   --jax_distributed.process_id=$PROCESS_ID
