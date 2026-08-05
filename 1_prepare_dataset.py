from utils import MCQ_TEMPLATE,init,load_tokenizer_and_model,get_prediction
import pandas as pd
from openai import OpenAI # 导入 OpenAI 库
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch
import torch.nn.functional as F
import json
import os
#from vllm import LLM, SamplingParams
import re
from tqdm import tqdm
import logging
import sys
#login(token="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")#huggingface login

#加载数据
def load_data(data_name,data_path,sub_set):
    login(token="hf_XdCKtEIjCScGfXHmnFBIcKzFzeBYEsPmI")
    if data_name in ['gpqa','mmlu','mmlu_redux']:
        data=load_dataset(data_path,sub_set)
    else:
        data=pd.read_csv(data_path)
    return data

#处理数据(MMLU_Redux)保证数据对应正确答案
def handle_data(data):
    data_test=data['test'].to_pandas()
    data_new=[]
    for i in range(len(data_test)):
        sample=data_test.iloc[i]
        if sample['error_type']=='ok':
            data_new.append({'question':sample['question'],'choices':'|'.join(sample['choices']),'answer':str(sample['answer'])})
        elif sample['error_type']=='wrong_groundtruth':
            if sample['correct_answer'] in ['0','1','2','3']:
                data_new.append({'question':sample['question'],'choices':'|'.join(sample['choices']),'answer':str(sample['correct_answer'])})
    return data_new

#think
def model_output(model,tokenizer,model_name,MCQ_TEMPLATE,data,think=True):
    if 'Qwen3' in model_name:
        model.eval()
        #option_ids=tokenizer.convert_tokens_to_ids(['A','B','C','D'])
        record_list=[]
        for i in tqdm(range(len(data)),ncols=80, ascii=True):
            sample=data[i]
            question=sample['question']
            choices=sample['choices'].split('|')
            A_text=choices[0]
            B_text=choices[1]
            C_text=choices[2]
            D_text=choices[3]
            MCQ=MCQ_TEMPLATE.format(q=question,o1=A_text,o2=B_text,o3=C_text,o4=D_text)
            if think:
                prompt=tokenizer.apply_chat_template(
                        [{"role":"user","content":MCQ}],
                        tokenize=False,#变成ids
                        add_generation_prompt=True,#增加assistance
                        enable_thinking=True,
                    )
            else:
                prompt=tokenizer.apply_chat_template(
                [{"role":"user","content":MCQ}],
                tokenize=False,#变成ids
                add_generation_prompt=True,#增加assistance
                enable_thinking=False,
            )
            #INPUT_TEMPLATES[model_name].format(user=MCQ,assistant='')
            inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs=model.generate(**inputs, **generation_args)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            all_output_text=output_text
            output_text = output_text[len(prompt):]#获取没有prompt的输出
            think_match = re.search(r'^(.*?)</think>', output_text, re.DOTALL)#提取</think>前的内容
            think_text = think_match.group(1).strip() if think_match else ""
            output_match = re.search(r'</think>\s*(.*)', output_text, re.DOTALL)#提取</think>后的内容
            output_text_wo_think=output_match.group(1).strip() if output_match else ""
            if think:
                match = re.findall(r"\\boxed\{([^}]+)\}", output_text,re.DOTALL)
            else:
                match = re.findall(r"\\boxed\{([^}]+)\}", output_text,re.DOTALL)
            if len(match)>=1:
                model_prediction = get_prediction(match[-1])
            else:
                model_prediction  = None
            ground_truth={'0':'A','1':'B','2':'C','3':'D'}[sample['answer']]
            correct=1 if ground_truth==model_prediction else 0
            if think:
                record_list.append({'CoT':think_text,'input_text':prompt,'output_text':output_text_wo_think,'all_output_text':all_output_text,'prediction':model_prediction,'truth':ground_truth,'correct':correct})
            else:
                record_list.append({'input_text':prompt,'output_text':output_text,'all_output_text':all_output_text,'prediction':model_prediction,'truth':ground_truth,'correct':correct})
        return record_list
#过滤得到think对nothink错的数据
def data_filter(think_data,nothink_data,origin_data):
    filter_data=[]
    for i in range(len(think_data)):
        
        if (think_data[i]['prediction']==None) or (nothink_data[i]['prediction']==None):
            continue
        if (think_data[i]['correct']==1) and (nothink_data[i]['correct']==0):
            filter_data.append({'question':origin_data[i]['question'],'choices':origin_data[i]['choices'],'CoT':think_data[i]['CoT'],
            'output_text':think_data[i]['output_text'],'prediction':think_data[i]['prediction'],'nothink_prediction':nothink_data[i]['prediction'],
            'truth':think_data[i]['truth'],'input_text':think_data[i]['input_text'],'all_output_text':think_data[i]['all_output_text']})
    return filter_data

if __name__ == '__main__':
    root="."
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
    #日志记录
    log_file = f"./log/{model_name}-prepare_dataset-{data_name}_{sub_set}.log"
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
    if os.path.isfile(f"{root}/datasets/{data_name}/{sub_set}"+"_new.json"):
        with open(f"{root}/datasets/{data_name}/{sub_set}"+"_new.json") as f:
            data_new=json.load(f)
            #data_new=data_new[:40]
    else:
        data=load_data(data_name,data_path,sub_set)
        os.makedirs(output_path, exist_ok=True)
        data_new=handle_data(data)
        #data_new=data_new[:40]
        #保存数据
        os.makedirs(f"{root}/datasets/{data_name}/", exist_ok=True)
        with open(f"{root}/datasets/{data_name}/{sub_set}"+"_new.json", 'w', encoding='utf-8') as f:
            json.dump(data_new,f, ensure_ascii=False, indent=4)
    #加载模型
    tokenizer,model=load_tokenizer_and_model(model_path)
    logging.info(f"finish load tokenizer and model({model_name})!")
    #模型生成参数
    generation_args = {
        "do_sample": False,
        "max_new_tokens": 32768,
        "use_cache": True,
        'repetition_penalty':1.2
    }
    record_list_think=model_output(model=model,tokenizer=tokenizer,model_name=model_name,MCQ_TEMPLATE=MCQ_TEMPLATE,data=data_new,think=True)
    #存储think模式的输出结果
    os.makedirs(f"{root}/results/{data_name}/", exist_ok=True)
    with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_think.json", 'w', encoding='utf-8') as f:
        json.dump(record_list_think,f, ensure_ascii=False, indent=4)
    logging.info("finish record think output")
    # log_dir = os.path.dirname(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_think.json")
    # os.makedirs(log_dir, exist_ok=True)
    with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_think.json") as f:
        record_list_think=json.load(f)
    record_list_nothink=model_output(model=model,tokenizer=tokenizer,model_name=model_name,MCQ_TEMPLATE=MCQ_TEMPLATE,data=data_new,think=False)
    #保存nothink模式的输出结果
    os.makedirs(f"{root}/results/{data_name}/", exist_ok=True)
    with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_nothink.json", 'w', encoding='utf-8') as f:
        json.dump(record_list_nothink,f, ensure_ascii=False, indent=4)
    logging.info("finish record nothink output")
    #保存filter_data
    filter_data=data_filter(think_data=record_list_think,nothink_data=record_list_nothink,origin_data=data_new)
    filter_data_len=len(filter_data)
    with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_filter_right.json", 'w', encoding='utf-8') as f:
        json.dump(filter_data,f, ensure_ascii=False, indent=4)
    logging.info("finish filter")
    total_num=len(record_list_think)
    think_none_num=0
    nothink_none_num=0
    for i in range(len(record_list_think)):
        if (record_list_think[i]['prediction']==None):
            think_none_num+=1
        if (record_list_nothink[i]['prediction']==None):
            nothink_none_num+=1
    logging.info(f"think none:{think_none_num}/{total_num},nothink none:{nothink_none_num}/{total_num}")
    logging.info(f"filter data: {filter_data_len}/{total_num}")
    #保存data
    # prepared_data=get_data(model=model,tokenizer=tokenizer,filter_data=filter_data,generation_args=generation_args)
    # data_len=len(prepared_data)
    # with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_prepared_right.json", 'w', encoding='utf-8') as f:
    #     json.dump(prepared_data,f, ensure_ascii=False, indent=4)
    # logging.info(f"finish get prepared data {data_len}")
    logging.info(f"finish prepare_dataset for {data_name}_{sub_set}_{model_name}")
