from utils import *
import json
import re
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import logging
import sys
import random
def maintain_only_sentences(all_sentences,indices):
    maintain_sentences=''
    for i in indices:
        maintain_sentences+=' ' 
        maintain_sentences+=all_sentences[i]
    return maintain_sentences

def get_maintain_sentences_results(maintain_sentences,text,ground_truth):
    #对text作拆解
    match_before = re.search(r'^(.*?)<think>', text, re.DOTALL)
    if match_before:
        before_think = match_before.group(0)
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL) 
    if match:
        CoT=match.group(1)
    after_think=' </think>'
    maintain_text=before_think+maintain_sentences+after_think
    prompt=maintain_text
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs=model.generate(**inputs, **generation_args)
                #logits = model(**inputs).logits
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        #output_text = output_text[len(prompt_wo_think):]#获取没有prompt的输出
        #print(output_text)
        think_match = re.search(r"<think>(.*?)</think>",maintain_text,re.DOTALL)#提取</think>前的内容
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
    return think_text,output_text_wo_think,model_prediction,correct,maintain_text

def calculate_percentage(sentence_att_list,indices,att_sink):
    value_sum=sum([sentence_att_list[idx] for idx in indices])
    percentage_in_all=value_sum/(1-att_sink)
    percentage_in_CoT=value_sum/sum(sentence_att_list)
    return value_sum,percentage_in_all,percentage_in_CoT

def get_record(sentence_list, indices, sentence_att_list, att_sink, filter_data_item):
    # 1. 筛选对应类别的句子
    maintain_sentences = maintain_only_sentences(sentence_list, indices)
    # 2. 计算注意力占比
    value_sum, percentage_in_all, percentage_in_CoT = calculate_percentage(
        sentence_att_list, indices, att_sink
    )
    # 3. 获取句子结果（CoT、预测、正确性等）
    think_text, output_text_wo_think, model_prediction, correct, maintain_text = get_maintain_sentences_results(
        maintain_sentences, filter_data_item['all_output_text'], filter_data_item['truth']
    )
    
    # 4. 构造并返回结果记录
    record = {
        'CoT': think_text,
        'output_text': output_text_wo_think,
        'prediction': model_prediction,
        'truth': filter_data_item['truth'],
        'correct': correct,
        'previous_prediction': filter_data_item['prediction'],
        'input_text_with_CoT': maintain_text,
        'att/grad_in_all': percentage_in_all,
        'att/grad_in_CoT': percentage_in_CoT,
        'att/grad_value': value_sum
    }
    return record

def save_record(record_list,output_path):
     with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(record_list, f, ensure_ascii=False, indent=4)


if __name__=='__main__':
    root='.'
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
      #日志记录
    log_file = f"./log/{model_name}-maintain_sentences-{data_name}_{sub_set}.log"
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
    with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_filter_right.json") as f:
        filter_data=json.load(f)#读取nothink错但think对的数据
    # with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_prepared_right.json") as f:
    #     prepared_data=json.load(f)
    #加载被保存的words
    with open(f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_words_list.json', encoding='utf-8') as f:
        words_list = json.load(f)
    with open(f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_all_att_list.json', encoding='utf-8') as f:
        all_att_list = json.load(f)
    top_k_percentage_list=[0.1,0.2,0.3] 
    for top_k_percentage in top_k_percentage_list:  
        record_list_high=[]
        record_list_low=[]
        record_list_random=[]
        record_list_random_wo_top_low=[]
        merge_result_list=[]
        for i in tqdm(range(len(filter_data))):
            sample=filter_data[i]
            question=sample['question']
            choices=sample['choices'].split('|')
            A_text=choices[0]
            B_text=choices[1]
            C_text=choices[2]
            D_text=choices[3]
            MCQ=MCQ_TEMPLATE.format(q=question,o1=A_text,o2=B_text,o3=C_text,o4=D_text)
            input_len=get_len(filter_data[i]['input_text'])
            CoT_match = re.search(r"<think>(.*?)</think>",filter_data[i]['all_output_text'], re.DOTALL)
            CoT_text = CoT_match.group(0) if CoT_match else ""
            explanation_match=re.search(r"</think>(.*?)\\boxed", filter_data[i]['all_output_text'], re.DOTALL) 
            explanation_text=explanation_match.group(1) if explanation_match else ""
            think_len=get_len(CoT_text)+get_len(explanation_text)
            total_att=all_att_list[i][input_len+1:(input_len+think_len-1)]
            total_words=words_list[i][input_len+1:(input_len+think_len-1)]
            word_att_tuple=list(zip(total_words,total_att))
            merge_result=merge_sentences_with_indices_v2(word_att_tuple)
            merge_result_list.append(merge_result)
            top_k=round(len(merge_result)*top_k_percentage)
            sentence_att_list=[x[1] for x in merge_result]
            sentence_list=[x[0] for x in merge_result]
            topk_indices=get_topk_indices_numpy(sentence_att_list,top_k=top_k)
            low_indices=get_low_indices_numpy(sentence_att_list,top_k=top_k)
            topk_indices=sorted(topk_indices)
            low_indices=sorted(low_indices)
            indices_for_random=[index for index in range(len(merge_result)) if ((index not in topk_indices) and (index not in low_indices))]
            random.seed(51)
            random_indices=sorted(random.sample(range(len(merge_result)),top_k))
            random_indices_wo_top_low=sorted(random.sample(indices_for_random,top_k))
            att_sink = all_att_list[i][0]
            #处理high类别
            high_record=get_record( sentence_list=sentence_list,
                                    indices=topk_indices,
                                    sentence_att_list=sentence_att_list,
                                    att_sink=att_sink,
                                    filter_data_item=sample)
            record_list_high.append(high_record)
            # 处理low类别
            low_record = get_record(
                sentence_list=sentence_list,
                indices=low_indices,
                sentence_att_list=sentence_att_list,
                att_sink=att_sink,
                filter_data_item=sample
            )
            record_list_low.append(low_record)
            # 处理random类别
            random_record = get_record(
                sentence_list=sentence_list,
                indices=random_indices,
                sentence_att_list=sentence_att_list,
                att_sink=att_sink,
                filter_data_item=sample,
            )
            record_list_random.append(random_record)
            #处理random_wo_top_low
            random_wo_top_low_record = get_record(
                sentence_list=sentence_list,
                indices=random_indices_wo_top_low,
                sentence_att_list=sentence_att_list,
                att_sink=att_sink,
                filter_data_item=sample,
            )
            record_list_random_wo_top_low.append(random_wo_top_low_record)
        output_path_high=f'{root}/final_results/maintain_only_sentences/high_att/'
        output_path_low=f'{root}/final_results/maintain_only_sentences/low_att/'
        output_path_random=f'{root}/final_results/maintain_only_sentences/random/'
        output_path_random_wo_top_low=f'{root}/final_results/maintain_only_sentences/random_wo_top_low/'
        os.makedirs(output_path_high, exist_ok=True)
        os.makedirs(output_path_low, exist_ok=True)
        os.makedirs(output_path_random, exist_ok=True)
        os.makedirs(output_path_random_wo_top_low, exist_ok=True)
        output_file=f"{model_name}-{data_name}-{sub_set}-att-right-top_percentage{top_k_percentage}.json"
        save_record(record_list_high,output_path_high+output_file)
        save_record(record_list_low,output_path_low+output_file)
        save_record(record_list_random,output_path_random+output_file)
        save_record(record_list_random_wo_top_low,output_path_random_wo_top_low+output_file)
        logging.info(f"save maintain sentences resultes successfully! top_k_percentage={top_k_percentage}")
    output_path_sentence=f'{root}/final_results/maintain_only_sentences/merge_result/'
    os.makedirs(output_path_sentence, exist_ok=True)
    output_file_sentence=f"{model_name}-{data_name}-{sub_set}-merge_result.json"
    with open(output_path_sentence+output_file_sentence, 'w', encoding='utf-8') as f:
        json.dump(merge_result_list,f, ensure_ascii=False, indent=4)
    logging.info(f"finish maintain sentences for {data_name}_{sub_set}_{model_name}")
