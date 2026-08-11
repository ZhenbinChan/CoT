CUDA_VISIABLE_DEVICES=1 python 4_maintain_only_topk_words.py \
	--data_name mmlu_redux  \
       	--sub_set global_facts  \
       	--model_name Qwen3-8B  \
       	--model_path /share/nlp/chenzhenbin/Workspaces/LLMs/Qwen3-8B
