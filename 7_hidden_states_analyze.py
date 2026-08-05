import torch
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import *
import os
import json
import logging
import sys
import umap
from scipy.stats import wasserstein_distance

def get_hidden_states(filter_data,model,tokenizer,generation_args):
    model.eval()
    all_hidden_states_last_list=[]
    all_hidden_states_pass_list=[]
    words_list=[]
    for idx in tqdm(range(len(filter_data))):
        text = filter_data[idx]['all_output_text']
        encoding = tokenizer(text, return_offsets_mapping=True)
        offsets = encoding['offset_mapping']
        with torch.no_grad():
            inputs=tokenizer(text,return_tensors='pt').to(model.device)
            hidden_states=model(**inputs,**generation_args).hidden_states
            encoding = tokenizer(text, return_offsets_mapping=True)
            offsets = encoding['offset_mapping']
            pieces = re.findall(r'(\S+|\s+)', text)#把字符串 text 拆成一个列表 pieces，
            #其中每个元素要么是一段连续的非空白字符（单词、数字、标点等），要么是一段连续的空白字符（空格、制表符 \t、换行 \n 等）
            words=[]
            word_spans = []
            start = 0
            for p in pieces:
                end = start + len(p)
                if not p.isspace():
                    words.append(p)                  # 只保留非空白
                    word_spans.append((start, end))
                start = end
            words_list.append(words) #获取对应单词
            hidden_states_last_list=[]#最后一层的hidden_states
            hidden_states_pass_list=[]#传给下一层的hidden_states
            for i,(start,end) in enumerate(word_spans):
                token_start, token_end=find_token_index(offsets=offsets,start_char= start,end_char= end)
                hidden_states_value=hidden_states[-2].detach().cpu().squeeze()[token_start:token_end+1].mean(dim=0)
                hidden_states_last_list.append(hidden_states_value)
                hidden_states_value=hidden_states[-1].detach().cpu().squeeze()[token_start:token_end+1].mean(dim=0)
                hidden_states_pass_list.append(hidden_states_value)
        all_hidden_states_last_list.append(hidden_states_last_list) 
        all_hidden_states_pass_list.append(hidden_states_pass_list)
    return  all_hidden_states_last_list,all_hidden_states_pass_list

def TSNE_analyze(filter_data,all_att_list,all_hidden_states_last_list,all_hidden_states_pass_list,top_k_percentage=0.1):
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    all_labels=[]
    all_tsne_2d_last=[]
    all_tsne_2d_pass=[]
    all_wasserstein_last_representativeness=[]
    all_wasserstein_pass_representativeness=[]
    # all_wasserstein_last_percentile=[]
    # all_wasserstein_pass_percentile=[]
    for i in tqdm(range(len(filter_data))):
        assert (len(all_att_list[i])==len(all_hidden_states_last_list[i])) and (len(all_att_list[i])==len(all_hidden_states_pass_list[i])), "length not match"
        input_len= len(filter_data[i]['input_text'].split())+1#加1保证后续取CoT的时候不包括<think>字符
        CoT_match = re.search(r"<think>(.*?)\\boxed",filter_data[i]['all_output_text'], re.DOTALL)#提取</think>前的内容
        CoT_text = CoT_match.group(1).strip() if CoT_match else ""
        CoT_len=len(CoT_text.split())#保证后续取CoT的时候不包括</think>字符
        top_k=round(CoT_len*top_k_percentage)
        topk_indices=get_topk_indices_numpy(all_att_list[i][input_len:(input_len+CoT_len)],top_k=top_k)
        labels=[0]*len(all_att_list[i])
        for j in topk_indices:
            labels[j+input_len]=1
        embeddings_2d_last = tsne.fit_transform(torch.stack(all_hidden_states_last_list[i]).numpy())
        embeddings_2d_pass =tsne.fit_transform(torch.stack(all_hidden_states_pass_list[i]).numpy())
        embeddings_2d_last_subset=[]
        embeddings_2d_pass_subset=[]
        for j in topk_indices:
            embeddings_2d_last_subset.append(embeddings_2d_last[j+input_len])
            embeddings_2d_pass_subset.append(embeddings_2d_pass[j+input_len])
        #draw_TSNE(embeddings_2d_last,labels,i,top_k_percentage,last_flag=True)
        #draw_TSNE(embeddings_2d_pass,labels,i,top_k_percentage,last_flag=False)
        wasserstein_last=calculate_wasserstein_distance(np.array(embeddings_2d_last),np.array(embeddings_2d_last_subset))
        wasserstein_pass=calculate_wasserstein_distance(np.array(embeddings_2d_pass),np.array(embeddings_2d_pass_subset))
        # result_last=advanced_representativeness(np.array(embeddings_2d_last),np.array(embeddings_2d_last_subset))
        # result_pass=advanced_representativeness(np.array(embeddings_2d_pass),np.array(embeddings_2d_pass_subset))
    
        all_labels.append(labels)
        all_tsne_2d_last.append(embeddings_2d_last)
        all_tsne_2d_pass.append(embeddings_2d_pass)
        all_wasserstein_last_representativeness.append(wasserstein_last)
        all_wasserstein_pass_representativeness.append(wasserstein_pass)
        # all_wasserstein_last_percentile.append(result_last['percentile'])
        # all_wasserstein_pass_percentile.append(result_pass['percentile'])
    return all_labels,all_tsne_2d_last,all_tsne_2d_pass,all_wasserstein_last_representativeness,all_wasserstein_pass_representativeness

def UMAP_analyze(filter_data,all_att_list,all_hidden_states_last_list,all_hidden_states_pass_list,top_k_percentage=0.1):
    reducer = umap.UMAP(n_neighbors=15,min_dist=0.1,n_components=2,random_state=42)
    all_labels=[]
    all_umap_2d_last=[]
    all_umap_2d_pass=[]
    all_wasserstein_last_representativeness=[]
    all_wasserstein_pass_representativeness=[]
    # all_wasserstein_last_percentile=[]
    # all_wasserstein_pass_percentile=[]
    for i in tqdm(range(len(filter_data))):
        assert (len(all_att_list[i])==len(all_hidden_states_last_list[i])) and (len(all_att_list[i])==len(all_hidden_states_pass_list[i])), "length not match"
        input_len= len(filter_data[i]['input_text'].split())+1#加1保证后续取CoT的时候不包括<think>字符
        CoT_match = re.search(r"<think>(.*?)\\boxed",filter_data[i]['all_output_text'], re.DOTALL)#提取</think>前的内容
        CoT_text = CoT_match.group(1).strip() if CoT_match else ""
        CoT_len=len(CoT_text.split())#保证后续取CoT的时候不包括</think>字符
        top_k=round(CoT_len*top_k_percentage)
        topk_indices=get_topk_indices_numpy(all_att_list[i][input_len:(input_len+CoT_len)],top_k=top_k)
        labels=[0]*len(all_att_list[i])
        for j in topk_indices:
            labels[j+input_len]=1
        embeddings_2d_last = reducer.fit_transform(torch.stack(all_hidden_states_last_list[i]).numpy())
        embeddings_2d_pass =reducer.fit_transform(torch.stack(all_hidden_states_pass_list[i]).numpy())
        embeddings_2d_last_subset=[]
        embeddings_2d_pass_subset=[]
        for j in topk_indices:
            embeddings_2d_last_subset.append(embeddings_2d_last[j+input_len])
            embeddings_2d_pass_subset.append(embeddings_2d_pass[j+input_len])
        #draw_TSNE(embeddings_2d_last,labels,i,top_k_percentage,last_flag=True)
        #draw_TSNE(embeddings_2d_pass,labels,i,top_k_percentage,last_flag=False)
        wasserstein_last=calculate_wasserstein_distance(np.array(embeddings_2d_last),np.array(embeddings_2d_last_subset))
        wasserstein_pass=calculate_wasserstein_distance(np.array(embeddings_2d_pass),np.array(embeddings_2d_pass_subset))
        # result_last=advanced_representativeness(np.array(embeddings_2d_last),np.array(embeddings_2d_last_subset))
        # result_pass=advanced_representativeness(np.array(embeddings_2d_pass),np.array(embeddings_2d_pass_subset))
    
        all_labels.append(labels)
        all_umap_2d_last.append(embeddings_2d_last)
        all_umap_2d_pass.append(embeddings_2d_pass)
        all_wasserstein_last_representativeness.append(wasserstein_last)
        all_wasserstein_pass_representativeness.append(wasserstein_pass)
        # all_wasserstein_last_percentile.append(result_last['percentile'])
        # all_wasserstein_pass_percentile.append(result_pass['percentile'])
    return all_labels,all_umap_2d_last,all_umap_2d_pass,all_wasserstein_last_representativeness,all_wasserstein_pass_representativeness



def draw_TSNE(embeddings_2d,labels,count,top_k_percentage,last_flag):
    data_name,_,sub_set,model_name,_,_=init()
    if last_flag:
        output_path=f"./tsne_pictures/{data_name}/{sub_set}/last/{model_name}/"
    else:
        output_path=f"./tsne_pictures/{data_name}/{sub_set}/pass/{model_name}/"
    title=f"topk{top_k_percentage}_TSNE_Visualization{count}"
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_path+title+".pdf", format='pdf', bbox_inches='tight')
    plt.close()

def calculate_wasserstein_distance(full_set,sub_set):
    wdim1 = wasserstein_distance(full_set[:, 0], sub_set[:, 0])
    wdim2 = wasserstein_distance(full_set[:, 1], sub_set[:, 1])
    wasserstein = (wdim1 + wdim2) / 2  # 简单平均
    worst_case = wasserstein_distance(full_set[:, 0], [np.mean(full_set[:, 0])])
    worst_case += wasserstein_distance(full_set[:, 1], [np.mean(full_set[:, 1])])
    worst_case /= 2
    representativeness = max(0, 1 - (wasserstein / worst_case))
    return representativeness

def advanced_representativeness(full_set, sub_set, n_samples=10):
    
    # 计算当前子集的距离
    wd_x = wasserstein_distance(full_set[:, 0], sub_set[:, 0])
    wd_y = wasserstein_distance(full_set[:, 1], sub_set[:, 1])
    current_wd = (wd_x + wd_y) / 2
    
    # 生成多个随机子集作为基准
    sample_size = len(sub_set)
    random_wds = []
    
    for _ in range(n_samples):
        # 随机采样
        random_indices = np.random.choice(len(full_set), sample_size, replace=False)
        random_subset = full_set[random_indices]
        
        # 计算随机采样的距离
        wd_x_rand = wasserstein_distance(full_set[:, 0], random_subset[:, 0])
        wd_y_rand = wasserstein_distance(full_set[:, 1], random_subset[:, 1])
        random_wd = (wd_x_rand + wd_y_rand) / 2
        random_wds.append(random_wd)
    
    # 计算代表性分数
    mean_random_wd = np.mean(random_wds)
    std_random_wd = np.std(random_wds)
    
    # 方法1：相对于随机采样的改进程度
    improvement = (mean_random_wd - current_wd) / mean_random_wd
    
    # 方法2：使用百分位数
    percentile = np.sum(current_wd < np.array(random_wds)) / n_samples
    
    return {
        'representativeness_score': max(0, improvement),
        'percentile': percentile,
        'current_distance': current_wd,
        'mean_random_distance': mean_random_wd,
        'std_random_distance': std_random_wd
    }

if __name__=='__main__':
    root="."
    method={
    0:'tsne',
    1:'umap'
    }[1]
    rate=1
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
    #日志记录
    log_file = f"./log/{model_name}-hidden_states_analyze-{data_name}_{sub_set}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    #data=load_data(data_name,data_path,sub_set)
    os.makedirs(output_path, exist_ok=True)
    tokenizer,model=load_tokenizer_and_model(model_path=model_path,output_hidden_states=True)
    model.eval()
    model.gradient_checkpointing_enable()
    logging.info(f"finish load tokenizer and model({model_name})!")
    #模型生成参数
    generation_args = {
        "do_sample": False,
        "max_new_tokens":4096,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        'repetition_penalty':1.2
    }
    #读取数据
    with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_filter_right.json") as f:
        filter_data=json.load(f)
    with open(f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_all_att_list.json') as f:
        all_att_list=json.load(f)
    all_hidden_states_last_list_file=f'./tsne_data/hidden_states/{data_name}/{model_name}/{sub_set}_rate{rate}_all_hidden_states_last_list.pth'
    all_hidden_states_pass_list_file=f'./tsne_data/hidden_states/{data_name}/{model_name}/{sub_set}_rate{rate}_all_hidden_states_pass_list.pth'
    if os.path.exists(all_hidden_states_last_list_file) and os.path.exists(all_hidden_states_pass_list_file):
        all_hidden_states_last_list=torch.load(all_hidden_states_last_list_file)
        all_hidden_states_pass_list=torch.load(all_hidden_states_pass_list_file)
    else:
        all_hidden_states_last_list,all_hidden_states_pass_list=get_hidden_states(filter_data,model,tokenizer,generation_args)
        os.makedirs(f'{root}/tsne_data/hidden_states/{data_name}/{model_name}/', exist_ok=True)
        torch.save(all_hidden_states_last_list, f'{root}/tsne_data/hidden_states/{data_name}/{model_name}/{sub_set}_rate{rate}_all_hidden_states_last_list.pth')
        torch.save(all_hidden_states_pass_list, f'{root}/tsne_data/hidden_states/{data_name}/{model_name}/{sub_set}_rate{rate}_all_hidden_states_pass_list.pth')
        logging.info(f"finish saving hidden_states data ({model_name}-{data_name}-{sub_set})!")
    topk_percentage_list=[0.1,0.2,0.3]
    os.makedirs(f'{root}/{method}_data/intermediate_result/{data_name}/{model_name}/', exist_ok=True)
    for top_k_percentage in topk_percentage_list:
        if method=='tsne':
            all_labels,all_2d_last,all_2d_pass,all_wasserstein_last_representativeness,all_wasserstein_pass_representativeness=TSNE_analyze(filter_data,all_att_list,all_hidden_states_last_list,all_hidden_states_pass_list,top_k_percentage)
        elif method=='umap':
            all_labels,all_2d_last,all_2d_pass,all_wasserstein_last_representativeness,all_wasserstein_pass_representativeness=UMAP_analyze(filter_data,all_att_list,all_hidden_states_last_list,all_hidden_states_pass_list,top_k_percentage)
        torch.save(all_labels,f'{root}/{method}_data/intermediate_result/{data_name}/{model_name}/{sub_set}_rate{rate}_all_labels_topk{top_k_percentage}.pth')
        torch.save(all_2d_last,f'{root}/{method}_data/intermediate_result/{data_name}/{model_name}/{sub_set}_rate{rate}_all_{method}_2d_last_topk{top_k_percentage}.pth')
        torch.save(all_2d_pass,f'{root}/{method}_data/intermediate_result/{data_name}/{model_name}/{sub_set}_rate{rate}_all_{method}_2d_pass_topk{top_k_percentage}.pth')
        with open(f'{root}/{method}_data/intermediate_result/{data_name}/{model_name}/{sub_set}_rate{rate}_last_representativeness{top_k_percentage}.json','w',encoding='utf-8') as file:
                json.dump(all_wasserstein_last_representativeness,file, ensure_ascii=False, indent=4)
        with open(f'{root}/{method}_data/intermediate_result/{data_name}/{model_name}/{sub_set}_rate{rate}_pass_representativeness{top_k_percentage}.json','w',encoding='utf-8') as file:
                json.dump(all_wasserstein_pass_representativeness,file, ensure_ascii=False, indent=4)
        # with open(f'{root}/tsne_data/intermediate_result/{data_name}/{model_name}_{sub_set}_rate{rate}_last_percentile{top_k_percentage}.json','w',encoding='utf-8') as file:
        #         json.dump(all_wasserstein_last_percentile,file, ensure_ascii=False, indent=4)
        # with open(f'{root}/tsne_data/intermediate_result/{data_name}/{model_name}_{sub_set}_rate{rate}_pass_percentile{top_k_percentage}.json','w',encoding='utf-8') as file:
        #         json.dump(all_wasserstein_pass_percentile,file, ensure_ascii=False, indent=4)
        logging.info(f'top_k_percentage={top_k_percentage},last representativeness={sum(all_wasserstein_last_representativeness)/len(all_wasserstein_last_representativeness):.2f}')
        logging.info(f'top_k_percentage={top_k_percentage},pass representativeness={sum(all_wasserstein_pass_representativeness)/len(all_wasserstein_pass_representativeness):.2f}')
        # logging.info(f'top_k_percentage={top_k_percentage},last percentile={sum(all_wasserstein_last_percentile)/len(all_wasserstein_last_percentile)}')
        # logging.info(f'top_k_percentage={top_k_percentage},pass percentile={sum(all_wasserstein_pass_percentile)/len(all_wasserstein_pass_percentile)}')
        logging.info(f'finish save intermediate result! top_k_percentage={top_k_percentage}')
    logging.info(f"finish hidden_states analyze for {data_name}_{sub_set}_{model_name}")