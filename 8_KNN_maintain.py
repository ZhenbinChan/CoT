from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import torch
from utils import *
import json
from tqdm import tqdm
import logging
import sys
import os
from sklearn.manifold import TSNE
import warnings
def find_k_nearest_sklearn(points_array, target_point, k=5):
    """
    使用scikit-learn找到最近的k个点
    """
    # 创建最近邻模型
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points_array)
    
    # 找到最近的k个点
    distances, indices = nbrs.kneighbors([target_point])
    sort_indices=sorted(indices[0])
    return points_array[indices[0]], distances[0],sort_indices
def get_maintain_results(replace_words,text,ground_truth,mask_flag):
    replace_text=maintain_only_topk(text,replace_words=replace_words,mask=mask_flag)
    prompt=replace_text
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs=model.generate(**inputs, **generation_args)
                #logits = model(**inputs).logits
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        #output_text = output_text[len(prompt_wo_think):]#获取没有prompt的输出
        #print(output_text)
        think_match = re.search(r"<think>(.*?)</think>",replace_text, re.DOTALL)#提取</think>前的内容
        think_text = think_match.group(0).strip() if think_match else ""
        output_match = re.search(r'</think>\s*(.*)', output_text, re.DOTALL)#提取</think>后的内容
        output_text_wo_think=output_match.group(1).strip() if output_match else ""
        match = re.findall(r"\\boxed\{([^}]+)\}", output_text,re.DOTALL)
        if len(match)>1:
            model_prediction = get_prediction(match[-1])
        else:
            model_prediction  = None
            #print("synonyms:",model_prediction)
        correct=1 if model_prediction==ground_truth else 0
    return think_text,output_text_wo_think,model_prediction,correct,replace_text
def get_maintain_KNN_results(data,filter_data):
    random.seed(51)
    record_list=[]
    indices_list=[]
    for i in tqdm(range(len(data))):
        points = data[i]
        target = random.choice(data[i])
        input_len=get_len(filter_data[i]['input_text'])
        CoT_match = re.search(r"<think>(.*?)</think>",filter_data[i]['all_output_text'], re.DOTALL)
        CoT_text = CoT_match.group(0) if CoT_match else ""
        explanation_match=re.search(r"</think>(.*?)\\boxed", filter_data[i]['all_output_text'], re.DOTALL) 
        explanation_text=explanation_match.group(1) if explanation_match else ""
        think_len=get_len(CoT_text)+get_len(explanation_text)
        top_k=round(think_len*top_k_percentage)
        # 查找最近点
        nearest_points, distances,indices = find_k_nearest_sklearn(points, target, top_k)
        assert len(words_list[i])==len(data_last[i])
        maintain_words=[words_list[i][j] for j in indices if (j>=input_len and j<input_len+think_len)]#获取近邻点对应的word
        maintain_att=[all_att_list[i][j] for j in indices if (j>=input_len and j<input_len+think_len)]#获取近邻点对应的attention
        #过滤会直接暴露答案的词
        for idx,word in enumerate(maintain_words):
            if ('A' in word) or ('B' in word) or ('C' in word) or ('D' in word):
                del maintain_words[idx]
                del maintain_att[idx]
        think_text,output_text_wo_think,model_prediction,correct,replace_text=get_maintain_results(maintain_words,filter_data[i]['all_output_text'],filter_data[i]['truth'],mask_flag=False)
        att_sink=all_att_list[i][0]
        value_sum=sum(maintain_att)#因为直接从思考过程中选择词，无需考虑att_sink
        percentage_in_all=value_sum/(1-att_sink)
        percentage_in_CoT=value_sum/sum(all_att_list[i][input_len:(input_len+think_len)])
        record_list.append({'CoT':think_text,'output_text':output_text_wo_think,'prediction':model_prediction,'truth':filter_data[i]['truth'],
        'correct':correct,'previous_prediction':filter_data[i]['prediction'],
        'input_text_with_CoT':replace_text,'att/grad_in_all':percentage_in_all,
        'att/grad_in_CoT':percentage_in_CoT,'att/grad_value':value_sum})
        indices_list.append(indices)
    return record_list,indices_list
# 示例使用
if __name__ == "__main__":
    # 方法A：屏蔽所有UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    root="."
    method={
    0:'tsne',
    1:'umap'
    }[1]
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
    log_file = f"./log/{model_name}-KNN_maintain-{data_name}_{sub_set}.log"
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    #加载模型
    tokenizer,model=load_tokenizer_and_model(model_path=model_path)
    logging.info(f"finish load tokenizer and model({model_name})!")
    #模型生成参数
    generation_args = {
        "do_sample": False,
        "max_new_tokens": 4096,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        'repetition_penalty':1.2
    }
    rate=1
    topk_percentage_list=[0.1,0.2,0.3]
    words_list_file=f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_words_list.json'
    all_att_list_file=f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_all_att_list.json'
    filter_data_file=f'{root}/results/{data_name}/{model_name}_{sub_set}_filter_right.json'

    with open(words_list_file)as f:
        words_list=json.load(f)
    with open(all_att_list_file) as f:
        all_att_list = json.load(f)
    with open(filter_data_file) as f:
        filter_data = json.load(f)
    for top_k_percentage in topk_percentage_list:
        labels_file=f'{root}/{method}_data/intermediate_result/{data_name}/{model_name}/{sub_set}_rate{rate}_all_labels_topk{top_k_percentage}.pth'
        last_file=f'{root}/{method}_data/intermediate_result/{data_name}/{model_name}/{sub_set}_rate{rate}_all_{method}_2d_last_topk{top_k_percentage}.pth'
        pass_file=f'{root}/{method}_data/intermediate_result/{data_name}/{model_name}/{sub_set}_rate{rate}_all_{method}_2d_pass_topk{top_k_percentage}.pth'
        labels=torch.load(labels_file,weights_only=False)
        data_last=torch.load(last_file,weights_only=False)
        data_pass=torch.load(pass_file,weights_only=False)
        # 生成数据
        record_list_last,indices_list_last=get_maintain_KNN_results(data_last,filter_data)
        logging.info(f'finish get last result! top_k_percentage={top_k_percentage}')
        record_list_pass,indices_list_pass=get_maintain_KNN_results(data_pass,filter_data)
        logging.info(f'finish get pass result! top_k_percentage={top_k_percentage}')
        output_path=f'{root}/final_results/maintain_KNN_{method}/filter/'
        mask_path='unmask/'
        os.makedirs(output_path+mask_path, exist_ok=True)
        output_file_last=f"{model_name}-{data_name}-{sub_set}-knn-last-top_percentage{top_k_percentage}.json"
        output_file_pass=f"{model_name}-{data_name}-{sub_set}-knn-pass-top_percentage{top_k_percentage}.json"
        output_indices_last=f"{model_name}-{data_name}-{sub_set}-indices-last-top_percentage{top_k_percentage}.pth"
        output_indices_pass=f"{model_name}-{data_name}-{sub_set}-indices-pass-top_percentage{top_k_percentage}.pth"
        with open(output_path+mask_path+output_file_last, 'w', encoding='utf-8') as f:
                json.dump(record_list_last, f, ensure_ascii=False, indent=4)
        with open(output_path+mask_path+output_file_pass, 'w', encoding='utf-8') as f:
                json.dump(record_list_pass, f, ensure_ascii=False, indent=4)
        torch.save(indices_list_last,output_path+mask_path+output_indices_last)
        torch.save(indices_list_pass,output_path+mask_path+output_indices_pass)
        logging.info(f'finish save result! top_k_percentage={top_k_percentage}')
    logging.info(f"finish KNN maintain for {data_name}_{sub_set}_{model_name}")
