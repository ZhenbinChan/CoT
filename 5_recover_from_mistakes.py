from utils import init,load_tokenizer_and_model,get_prediction,InsertTokenProcessor
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import os
import logging
import sys
from transformers import StoppingCriteria, StoppingCriteriaList
def recover_from_mistakes(model,tokenizer,data):
    record_list=[]
    for i in tqdm(range(len(data))):
        input_text=data[i]['input_text_with_CoT'].replace('</think>','')
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs=model.generate(**inputs, **generation_args,logits_processor=[insert_processor])
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            think_match = re.search(r"<think>(.*?)</think>",output_text, re.DOTALL)#提取</think>前的内容
            think_text = think_match.group(0).strip() if think_match else ""
            output_match = re.search(r'</think>\s*(.*)', output_text, re.DOTALL)#提取</think>后的内容
            output_text_wo_think=output_match.group(1).strip() if output_match else ""
            match = re.findall(r"\\boxed\{([^}]+)\}", output_text,re.DOTALL)
            if len(match)>1:
                model_prediction = get_prediction(match[-1])
            else:
                model_prediction  = None
                #print("synonyms:",model_prediction)
            correct=1 if model_prediction==data[i]['truth'] else 0
        record_list.append({'CoT':think_text,'output_text_wo_think':output_text_wo_think,'prediction':model_prediction,'truth':data[i]['truth'],
                        'correct':correct,'previous_prediction':data[i]['previous_prediction'],'replaced_phrases':data[i]['replaced_phrases'],
                        'antonyms':data[i]['antonyms'],'all_output':output_text})
    return record_list
if __name__=='__main__':
    root="."
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
    log_file = f"./log/{model_name}-recover_from_mistakes-{data_name}_{sub_set}.log"
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
    tokenizer,model=load_tokenizer_and_model(model_path)
    logging.info(f"finish load tokenizer and model({model_name})!")
    #插入词
    insert_processor = InsertTokenProcessor(
        insert_token_id=tokenizer.encode("</think>", add_special_tokens=False)[0],
        insert_position=8192)
    generation_args = {
        "do_sample": False,
        "max_new_tokens": 10000,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        'repetition_penalty':1.2
    }
    rate=1
    top_k_percentage_list=[0.1,0.2,0.3]
    kind_list=['random','attention']
    for kind in kind_list:
        for top_k_percentage in top_k_percentage_list:
            input_path=f'{root}/final_results/replace_results/'
            if kind=='random':
                input_file=f"{model_name}-{data_name}-{sub_set}-antonym-rate{rate}_random_right.json"
            elif kind=='attention':
                input_file=f"{model_name}-{data_name}-{sub_set}-antonym-att-right-top_percentage{top_k_percentage}.json"
            else:
                input_file=f"{model_name}-{data_name}-{sub_set}-antonym-grad-correct-right-top_percentage{top_k_percentage}.json"
            with open(input_path+input_file) as file:
                data=json.load(file)
            record_list=recover_from_mistakes(model,tokenizer,data)
            logging.info(f'get recover results successfully! kind={kind},top_k_percentage={top_k_percentage}')
            output_path=f'{root}/final_results/recover_results/'
            os.makedirs(output_path,exist_ok=True)
            output_file=input_file
            with open(output_path+output_file,'w',encoding='utf-8') as file:
                json.dump(record_list,file, ensure_ascii=False, indent=4)
    logging.info(f"finish recover from mistakes for {data_name}_{sub_set}_{model_name}")
