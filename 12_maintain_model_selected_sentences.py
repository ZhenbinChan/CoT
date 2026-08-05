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
def normalize(s):
    return ''.join(s.split())

def string_match(str1,str2):
    str1_normalize=normalize(str1)
    st2_normalize=normalize(str2)
    if (str1_normalize in st2_normalize) or (st2_normalize in str1_normalize):
        return True
    else:
        return False
if __name__=='__main__':
    root='.'
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
    #日志记录
    log_file = f"./log/{model_name}-maintain_model_selected_sentences-{data_name}_{sub_set}.log"
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
    with open(f'./final_results/metric/origin/{model_name}-{data_name}-{sub_set}-metric_record(rouge_L).json') as f:
        metric_record=json.load(f)
    with open(f'./final_results/maintain_only_sentences/merge_result/{model_name}-{data_name}-{sub_set}-merge_result.json') as f:
        merge_result=json.load(f)
    record_list=[]
    for i in tqdm(range(len(filter_data))):
        text=filter_data[i]['all_output_text']
        ground_truth=filter_data[i]['truth']
        model_prediction=[x[0] for x in metric_record[i]['model_prediction']]
        match_sentences=[]
        for r in range(len(merge_result[i])):
            for model_prediction_sample in model_prediction:
                if string_match(merge_result[i][r][0],model_prediction_sample):
                    match_sentences.append(merge_result[i][r][0]) 
        #match_sentences=[x[0] for x in merge_result[i] if x[0] in model_prediction]
        maintain_sentences=''
        for j in range(len(match_sentences)):
            maintain_sentences+=' ' 
            maintain_sentences+=match_sentences[j]
        think_text,output_text_wo_think,model_prediction,correct,maintain_text=get_maintain_sentences_results(maintain_sentences,text,ground_truth)
        record = {
        'CoT': think_text,
        'output_text': output_text_wo_think,
        'prediction': model_prediction,
        'truth': filter_data[i]['truth'],
        'correct': correct,
        'previous_prediction': filter_data[i]['prediction'],
        'input_text_with_CoT': maintain_text,
        'score':metric_record[i]['score']
        }
        record_list.append(record)
        output_path=f'{root}/final_results/maintain_only_sentences/model_select/'
        os.makedirs(output_path, exist_ok=True)
        output_file=f"{model_name}-{data_name}-{sub_set}-model_select.json"
        with open(output_path+output_file, 'w', encoding='utf-8') as f:
            json.dump(record_list, f, ensure_ascii=False, indent=4)
    logging.info(f"finish maintain selected sentences for {data_name}_{sub_set}_{model_name}")