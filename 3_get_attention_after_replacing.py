from utils import *
import json
import re
import torch
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch.nn.functional as F
import logging
import sys
#获取注意力和梯度
def get_att(model,tokenizer,data,generation_args):
    words_list=[]
    all_att_list=[]
    #all_grad_list_correct=[]
    all_CoT_att_list=[]
    #all_CoT_grad_list_correct=[]
    model.eval()
    for idx in tqdm(range(len(data))):
        CoT_start=None
        CoT_end=None
        input_tex_w_CoT=data[idx]['input_text_with_CoT']
        match_before = re.search(r'^(.*?)<think>', input_tex_w_CoT, re.DOTALL)
        if match_before:
            input_text = match_before.group(1)
        else:
            input_text=''
        #text=input_text+data[idx]['CoT']+data[idx]['output_text']
        match_after=re.search(r'\\boxed\s*(.*)', data[idx]['output_text'], re.DOTALL)
        if match_after:
            box_text=match_after.group(0)
        else:
            box_text=''
        text=input_text+data[idx]['CoT']+data[idx]['output_text_w_mark']+box_text
        encoding = tokenizer(text, return_offsets_mapping=True)
        offsets = encoding['offset_mapping']
        input_length=len(input_text)
        #寻找答案所在输出string的位置
        #print(text[input_length:])
        match = re.search(r"\\boxed\{([^}]+)\}", text[input_length:],re.DOTALL)
        if match:
            start=match.start()
            end=match.end()
        sub_text=text[:start+7+input_length]#截取到\box{停止，从而使得last_token为正确答案的前一个token
        with torch.no_grad():
            output_attention=get_attention(model,tokenizer,generation_args,sub_text)
            att = torch.stack(output_attention, dim=0).mean(dim=0)
            att_cpu = att.cpu()
            last_token_att = att_cpu.mean(dim=1).squeeze()
            del output_attention, att
            if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            match = re.search(r"\<think\>", text,re.DOTALL)
            if match:
                    CoT_start=match.start()
            match = re.search(r"\</think\>", text,re.DOTALL)
            if match:
                    CoT_end=match.end()
            CoT_end=len(text) #当将explanation也视为思考过程时，则启用这个
            token_start, token_end=find_token_index(offsets=offsets,start_char= CoT_start,end_char=CoT_end)
            #计算CoT attention占总attention的比例 (撇去第一个出现att sink现象的token)
            CoT_att=last_token_att[token_start:token_end+1].sum()
            all_CoT_att_list.append(CoT_att.item())
            #     CoT_att_percentage=CoT_att/(1-last_token_att[0])
            #     #计算被替换词att占CoT att的比例
            #     change_att_percentage_in_CoT=total_att/CoT_att
            #获取每一个单词的attention
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
            #answer_list=["A","B","C","D"]
            #grad
            # correct_answer=filter_data[idx]['truth']
            # change_token_id_correct = tokenizer.encode(correct_answer)[0]
            # analysis_correct =analyze_output_impact_gradients(model,tokenizer,sub_text,change_token_id_correct)
            # token_saliency_correct=analysis_correct['input_gradients'].squeeze(0).norm(dim=1)
            # CoT_grad=token_saliency_correct[token_start:token_end+1].sum()
            # all_CoT_grad_list_correct.append(CoT_grad.item())
            att_list=[]
            #grad_list_correct=[]
            #grad_list_uncorrect=[]
            for i,(start,end) in enumerate(word_spans):
                token_start, token_end=find_token_index(offsets=offsets,start_char= start,end_char= end)
                att_value=last_token_att[token_start:token_end+1].sum()
                att_list.append(att_value.item())
                #grad_value=token_saliency_correct[token_start:token_end+1].sum()
                #grad_list_correct.append(float(grad_value))#将float16转为原生float类型方便json写出
            all_att_list.append(att_list) #获取单词对应att
            #all_grad_list_correct.append(grad_list_correct)
    
    return all_att_list, all_CoT_att_list, words_list

#获取随机替换的最终结果
def get_random_results(random_data,words_list,all_att_list):
    record_list=[]
    for idx in tqdm(range(len(random_data))):
        indices = [i for i, item in enumerate(words_list[idx]) if '[' in item and ']' in item]
        replace_after_att=[all_att_list[idx][i] for i in indices]
        att_sum=sum(replace_after_att)
        att_percentage_in_CoT=att_sum/all_CoT_att_list[idx]
        new_data=random_data[idx]
        new_data['att_after_replacing']=att_sum
        new_data['att_percentage_in_CoT_after_replacing']=att_percentage_in_CoT
        record_list.append(new_data)
    return record_list

#获取替换topk的最终结果
def get_top_results(top_data,words_list,all_att_list):
    record_list=[]
    for idx in tqdm(range(len(top_data))):
        indices = [i for i, item in enumerate(words_list[idx]) if '[' in item and ']' in item]
        replace_after_att=[all_att_list[idx][i] for i in indices]
        att_sum=sum(replace_after_att)
        att_percentage_in_CoT=att_sum/all_CoT_att_list[idx]
        new_data=top_data[idx]
        new_data['att_after_replacing']=att_sum
        new_data['att_percentage_in_CoT_after_replacing']=att_percentage_in_CoT
        record_list.append(new_data)
        # else:
        #     replace_after_grad=[all_grad_list[idx][i] for i in indices]
        #     grad_sum=sum(replace_after_grad)
        #     grad_percentage_in_CoT=grad_sum/all_CoT_grad_list[idx]
        #     new_data=top_data[idx]
        #     new_data['grad_after_replacing']=grad_sum
        #     new_data['grad_percentage_in_CoT_after_replacing']=grad_percentage_in_CoT
        #     record_list.append(new_data)
    return record_list
if __name__=='__main__':
    root='.'
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
     #日志记录
    log_file = f"./log/{model_name}-get_attention_and_gradient_after_replacing-{data_name}_{sub_set}.log"
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
    generation_args = {
        "do_sample": False,
        "max_new_tokens":1024,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        'repetition_penalty':1.2
    }
    logging.info(f"finish load tokenizer and model({model_name})!")
    #加载数据
    with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_filter_right.json") as f:
        filter_data=json.load(f)
    #保存random中间结果
    logging.info(f"save random intermediate results!")
    rate=1
    output_path=f'{root}/final_results/replace_results/'
    output_file_synonyms=f"{model_name}-{data_name}-{sub_set}-synonym-rate{rate}_random_right.json"
    output_file_antonyms=f"{model_name}-{data_name}-{sub_set}-antonym-rate{rate}_random_right.json"
    with open(output_path+output_file_synonyms) as f:
        random_synonyms_data=json.load(f)
    with open(output_path+output_file_antonyms) as f:
        random_antonyms_data=json.load(f)
    os.makedirs(f'{root}/results/{data_name}/att_grad_after/', exist_ok=True)
    kind_list=['synonyms','antonyms']
    for kind in kind_list:
        if kind == 'synonyms':
            random_data=random_synonyms_data
        else:
            random_data=random_antonyms_data
        all_att_list, all_CoT_att_list, words_list=get_att(model,tokenizer,random_data,generation_args=generation_args)  
        logging.info("finish getting att and grad after for random data")  
        with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_random_rate{rate}_all_att_list_{kind}.json', 'w', encoding='utf-8') as f:
            json.dump(all_att_list, f, ensure_ascii=False, indent=4)
        with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_random_rate{rate}_all_CoT_att_list_{kind}.json', 'w', encoding='utf-8') as f:
            json.dump(all_CoT_att_list, f, ensure_ascii=False, indent=4)
        #保存grad_list
        # with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_random_rate{rate}_all_grad_list_correct_{kind}.json', 'w', encoding='utf-8') as f:
        #     json.dump(all_grad_list_correct, f, ensure_ascii=False, indent=4)
        # with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_random_rate{rate}_all_CoT_grad_list_correct_{kind}.json', 'w', encoding='utf-8') as f:
        #     json.dump(all_CoT_grad_list_correct, f, ensure_ascii=False, indent=4)
        #保存words_list
        with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_random_rate{rate}_words_list_{kind}.json', 'w', encoding='utf-8') as f:
            json.dump(words_list, f, ensure_ascii=False, indent=4)
        #保存random最终结果
        record_list=get_random_results(random_data,words_list,all_att_list)
        logging.info("get random results successfully!")
        if kind == 'synonyms':
            path=output_path+output_file_synonyms
        else:
            path=output_path+output_file_antonyms
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(record_list,f,ensure_ascii=False, indent=4)
    #保存att/grad中间结果
    logging.info("get att/grad intermediate result")
    #att_flag_list=[True,False]
    top_k_percentage_list=[0.1,0.2,0.3]
    #for att_flag in att_flag_list:
    for kind in kind_list:
        for top_k_percentage in top_k_percentage_list:
            output_file_synonyms=f"{model_name}-{data_name}-{sub_set}-synonym-att-right-top_percentage{top_k_percentage}.json"
            output_file_antonyms=f"{model_name}-{data_name}-{sub_set}-antonym-att-right-top_percentage{top_k_percentage}.json"
            # else:
            #     output_file_synonyms=f"{model_name}-{data_name}-{sub_set}-synonym-grad-correct-right-top_percentage{top_k_percentage}.json"
            #     output_file_antonyms=f"{model_name}-{data_name}-{sub_set}-antonym-grad-correct-right-top_percentage{top_k_percentage}.json"
            with open(output_path+output_file_synonyms) as f:
                synonyms_data=json.load(f)
            with open(output_path+output_file_antonyms) as f:
                antonyms_data=json.load(f)
            if kind == 'synonyms':
                top_data=synonyms_data
            else:
                top_data=antonyms_data
            all_att_list, all_CoT_att_list, words_list=get_att(model,tokenizer,top_data,generation_args=generation_args)
            logging.info("finish getting att and grad after for top data")     
            with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_att_top_percentage{top_k_percentage}_all_att_list_{kind}.json', 'w', encoding='utf-8') as f:
                json.dump(all_att_list, f, ensure_ascii=False, indent=4)
            with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_att_top_percentage{top_k_percentage}_all_CoT_att_list_{kind}.json', 'w', encoding='utf-8') as f:
                json.dump(all_CoT_att_list, f, ensure_ascii=False, indent=4)
            with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_att_top_percentage{top_k_percentage}_words_list_{kind}.json', 'w', encoding='utf-8') as f:
                json.dump(words_list, f, ensure_ascii=False, indent=4)
            #保存grad_list
            # else:
            #     with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_grad_top_percentage{top_k_percentage}_all_grad_list_correct_{kind}.json', 'w', encoding='utf-8') as f:
            #         json.dump(all_grad_list_correct, f, ensure_ascii=False, indent=4)
            #     with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_grad_top_percentage{top_k_percentage}_all_CoT_grad_list_correct_{kind}.json', 'w', encoding='utf-8') as f:
            #         json.dump(all_CoT_grad_list_correct, f, ensure_ascii=False, indent=4)
            #     with open(f'{root}/results/{data_name}/att_grad_after/{model_name}_{sub_set}_grad_top_percentage{top_k_percentage}_words_list_{kind}.json', 'w', encoding='utf-8') as f:
            #         json.dump(words_list, f, ensure_ascii=False, indent=4)
            logging.info(f"finish save topk data! kind={kind},top_k_percentage={top_k_percentage}")
            #保存最终结果
            record_list=get_top_results(top_data,words_list,all_att_list)
            logging.info("get top results successfully!")
            if kind == 'synonyms':
                path=output_path+output_file_synonyms
            else:
                path=output_path+output_file_antonyms
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(record_list,f,ensure_ascii=False, indent=4)
    logging.info(f"finish get attention after replacing for {data_name}_{sub_set}_{model_name}")
