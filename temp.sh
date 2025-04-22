python3 run_benchmark.py cuda-mem \
    -m alexnet,BERT_pytorch,resnet50,resnet152 \
    -b 64 -r 1 -t train 
    # -a "/home/chensichao/ptuvm/comparisons/simple_cudaMallocManaged/alloc.so" \
    # --alloc-func my_malloc \
    # --free-func my_free \
    # --cache_info-func my_cache_info


# alexnet,BERT_pytorch,dcgan,demucs,densenet121,dlrm,lennard_jones,llama,llama_v2_7b_16h,maml,maml_omniglot,microbench_unbacked_tolist_sum,mnasnet1_0,mobilenet_v2,mobilenet_v2_quantized_qat,moco

# basic_gnn_gcn,
# drq,fastNLP_Bert,hf_Albert,hf_Bart,
# llava
# mobilenet_v3_large