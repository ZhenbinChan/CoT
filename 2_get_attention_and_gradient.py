from utils import *
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
import random
import spacy
from tqdm import tqdm
import logging
import sys
#login(token="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")#huggingface login
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

#调用deepseek-v3作单词替换
def extract_and_replace_phrases(text, rate=1,type='Synonyms',base_url=None,api_key=None,model=None,replace_words=None):
    # 加载 spaCy 英文模型
    match_before = re.search(r'^(.*?)<think>', text, re.DOTALL)
    if match_before:
        before_think = match_before.group(1)
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL) 
    if match:
        CoT=match.group(0)
    match_explanation=re.search(r"</think>(.*?)\\boxed", text, re.DOTALL) 
    if match_explanation:
        explanation=match_explanation.group(1)
    if not replace_words:
        random.seed(51)#51
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            # 如果模型不存在，尝试下载
            from spacy.cli import download
            download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
        doc = nlp(CoT+explanation)
        #model.eval()
        # 抽取所有名词短语（noun chunks）
        #phrases = list(set(chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip().split()) > 1))
        # 抽取所有形容词+副词
        phrases = [token.text for token in doc if (token.pos_ == "ADJ" or token.pos_ == "ADV")]
        #如果包含数学公式，数学公式也会被选为形容词/副词
        phrases_deduplicate=list(set(phrases))#去重
        # 随机采样 n 个短语（如果短语数量不够，则取全部）
        sampled_phrases = random.sample(phrases_deduplicate, min(round(rate*len(phrases_deduplicate)), len(phrases_deduplicate)))

        #不去重
        # 随机采样 n 个短语（如果短语数量不够，则取全部）
        #sampled_phrases = random.sample(phrases, min(round(rate*len(phrases_deduplicate)), len(phrases_deduplicate)))

    else:
        sampled_list=replace_words
        sampled_list = sorted(sampled_list, key=lambda x: x[1], reverse=True)#倒序替换
        sampled_phrases=[word[0] for word in sampled_list ]
    if type=='Synonyms':
        query=Synonyms_TEMPLATE.format(q=sampled_phrases)
        _,output=call_ark(query,api_key, base_url, model)
    else:
        query=Antonyms_TEMPLATE.format(q=sampled_phrases)
        _,output=call_ark(query,api_key, base_url, model)
    #content_match = re.search(r'{(.*?)}', output, re.DOTALL)#提取正式答案{}部分的内容
    #content_inside = content_match.group(1).strip() if content_match else ""
    #上面那一步可能会导致提取到公式中的}从而过早终止
    lines = output.strip('{}\n').split(',')
    values_list = [line.split(':')[-1].strip().strip("'").strip('"').strip("'") for line in lines if ":" in line]
    CoT_text=CoT+explanation
    CoT_text_w_mark=CoT+explanation
    match_list=[]#被替换词的位置
    total_num=0#被替换的词数
    if not replace_words:#如果是随机替换，则把选中词全替换
        for j,phrase in enumerate(sampled_phrases):
            if(len(values_list)!=len(sampled_phrases)):
                if j==len(values_list):
                    break
            if (phrase in '<think>') or (phrase in '</think>'):#如果<think>或者</think>被选择为要被替换的词语则跳过不进行替换
                continue
            if (values_list[j] in '<think>') or (values_list[j] in '</think>'):#如果生成的替换词为<think>或者</think>则跳过不进行替换
                continue
            match_list.append([(m.start(),m.end()) for m in re.finditer(re.escape(phrase), text)])
            total_num+=CoT_text.count(phrase)
            CoT_text=CoT_text.replace(phrase,values_list[j])
            CoT_text_w_mark = CoT_text_w_mark.replace(phrase,'['+values_list[j]+']')
            #不去重版本
            # total_num+=1
            # CoT_text=CoT_text.replace(phrase,values_list[j],1)
            # CoT_text_w_mark = CoT_text_w_mark.replace(phrase,'['+values_list[j]+']',1)
        replace_text=before_think+CoT_text
        replace_text_w_mark=before_think+CoT_text_w_mark
    
    else:
        for j,phrase in enumerate(sampled_phrases):
            if(len(values_list)!=len(sampled_phrases)):
                if j==len(values_list):
                    break
            if (phrase in '<think>') or (phrase in '</think>'):#如果<think>或者</think>被选择为要被替换的词语则跳过不进行替换
                continue
            if (values_list[j] in '<think>') or (values_list[j] in '</think>'):#如果生成的替换词为<think>或者</think>则跳过不进行替换
                continue
            total_num+=1
            CoT_text=replace_nth(CoT_text,phrase,values_list[j],sampled_list[j][1])
            CoT_text_w_mark = replace_nth(CoT_text_w_mark,phrase,'['+values_list[j]+']',sampled_list[j][1])
        replace_text=before_think+CoT_text
        replace_text_w_mark=before_think+CoT_text_w_mark
    percentage=total_num*1.0/len(CoT.split(' '))
    match = re.search(r"(.*?)</think>", replace_text, re.DOTALL) 
    if not match:
        replace_text=None
    match = re.search(r"(.*?)</think>", replace_text_w_mark, re.DOTALL) 
    if not match:
        replace_text_w_mark=None   
    # match = re.search(r"(.*?)</think>", replace_text, re.DOTALL) 
    # if match:
    #     replace_text=match.group(0)
    # else:
    #     replace_text=None
    # match = re.search(r"(.*?)</think>", replace_text_w_mark, re.DOTALL) 
    # if match:
    #     replace_text_w_mark=match.group(0)
    # else:
    #     replace_text_w_mark=None
    return replace_text, sampled_phrases,values_list,replace_text_w_mark,match_list,percentage

#获取注意力和梯度
def get_att(filter_data,model,tokenizer,generation_args,rate=1):
    base_url,api_key,api_model_name=init_api()
    save_list=[]
    words_list=[]
    all_att_list=[]
    model.eval()
    # all_grad_list_correct=[]
    # all_grad_list_uncorrect=[]
    for idx in tqdm(range(len(filter_data))):
        text = filter_data[idx]['all_output_text']
        encoding = tokenizer(text, return_offsets_mapping=True)
        offsets = encoding['offset_mapping']
        input_length=len(filter_data[idx]['input_text'])
        #寻找答案所在输出string的位置
        match = re.search(r"\\boxed\{([^}]+)\}", text[len(filter_data[idx]['input_text']):],re.DOTALL)
        if match:
            start=match.start()
            end=match.end()
        sub_text=text[:start+7+input_length]#截取到\box{停止，从而使得last_token为正确答案的前一个token
        #inputs = tokenizer(sub_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_attention=get_attention(model,tokenizer,generation_args,sub_text)
            att = torch.stack(output_attention, dim=0).mean(dim=0)
            att_cpu = att.cpu()
            last_token_att = att_cpu.mean(dim=1).squeeze()
            del output_attention, att
            if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            replace_text, replaced_phrases, synonyms,replace_text_w_mark,matches_list,percentage = extract_and_replace_phrases(filter_data[idx]['all_output_text'], rate=rate,type='Synonyms',
        base_url=base_url,api_key=api_key,model=api_model_name)
            
            #获取被替换词在token_id的位置
            token_position_list=[]
            for sample in matches_list:
                t_list=[]
                for position in sample:
                    start_char = position[0]
                    end_char = position[1]
                    token_start, token_end=find_token_index(offsets=offsets,start_char=start_char,end_char=end_char)
                    t_list.append((token_start,token_end))
                token_position_list.append(t_list)
            #if get_att:
            #计算att
            total_att_list=[]#计算被替换词的att权重和
            total_att=0
            for sample in token_position_list:
                sample_att_list=[]
                for position in sample:
                    token_start=position[0]
                    token_end=position[1]
                    sample_att_list.append(last_token_att[token_start:token_end+1].sum())
                    total_att+=last_token_att[token_start:token_end+1].sum()
                total_att_list.append(sample_att_list)
            match = re.search(r"\<think\>", text,re.DOTALL)
            if match:
                    CoT_start=match.start()
            match = re.search(r"\</think\>", text,re.DOTALL)
            if match:
                    CoT_end=match.end()
            CoT_end=len(text)
            token_start, token_end=find_token_index(offsets=offsets,start_char= CoT_start,end_char=CoT_end)
            #计算CoT attention占总attention的比例 (撇去第一个出现att sink现象的token)
            CoT_att=last_token_att[token_start:token_end+1].sum()
            CoT_att_percentage=CoT_att/(1-last_token_att[0])
            #计算被替换词att占CoT att的比例
            change_att_percentage_in_CoT=total_att/CoT_att
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
        #if get_grad:#获取梯度
        # answer_list=["A","B","C","D"]
        # correct_answer=filter_data[idx]['truth']
        # index = answer_list.index(correct_answer)
        # uncorrect_index=(index+1)%4
        # uncorrect_answer=answer_list[uncorrect_index]
        # change_token_id_correct = tokenizer.encode(correct_answer)[0]
        # analysis_correct =analyze_output_impact_gradients(model,tokenizer,sub_text,change_token_id_correct)
        # token_saliency_correct=analysis_correct['input_gradients'].squeeze(0).norm(dim=1)
        # change_token_id_uncorrect = tokenizer.encode(uncorrect_answer)[0]
        # analysis_uncorrect =analyze_output_impact_gradients(model,tokenizer,sub_text,change_token_id_uncorrect)
        # token_saliency_uncorrect=analysis_uncorrect['input_gradients'].squeeze(0).norm(dim=1)
            att_list=[]
        # grad_list_correct=[]
        # grad_list_uncorrect=[]
            for i,(start,end) in enumerate(word_spans):
                token_start, token_end=find_token_index(offsets=offsets,start_char= start,end_char= end)
            #if get_att:
                att_value=last_token_att[token_start:token_end+1].sum()
                att_list.append(att_value.item())
            del last_token_att
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            #if get_grad:
            # grad_value=token_saliency_correct[token_start:token_end+1].sum()
            # grad_list_correct.append(float(grad_value))#将float16转为原生float类型方便json写出
            # grad_value=token_saliency_uncorrect[token_start:token_end+1].sum()
            # grad_list_uncorrect.append(float(grad_value))#将float16转为原生float类型方便json写出
            all_att_list.append(att_list) #获取单词对应att
        # all_grad_list_correct.append(grad_list_correct)
        # all_grad_list_uncorrect.append(grad_list_uncorrect)
        #if get_att:
            save_list.append({'change_attention':total_att.item(),'CoT_att_percentage':CoT_att_percentage.item(),'change_attention_percentage_in_CoT': change_att_percentage_in_CoT.item()})
    return save_list,all_att_list,words_list


#进行随机替换
def random_replace(model,model_name,filter_data,att_data,synonyms_flag=True,rate=1):
    base_url,api_key,api_model_name=init_api()
    #获得
    record_list=[]
    for i in tqdm(range(len(filter_data))):
        # if i >0:
        #     break
        if 'Qwen3' in model_name:
            random.seed(51)#51
            model.eval()
            sample=filter_data[i]
            question=sample['question']
            choices=sample['choices'].split('|')
            A_text=choices[0]
            B_text=choices[1]
            C_text=choices[2]
            D_text=choices[3]
            MCQ=MCQ_TEMPLATE.format(q=question,o1=A_text,o2=B_text,o3=C_text,o4=D_text)
        if synonyms_flag:
            replace_text, replaced_phrases, synonyms,replace_text_w_mark,matches_list,percentage = extract_and_replace_phrases(filter_data[i]['all_output_text'], rate=rate,type='Synonyms',
        base_url=base_url,api_key=api_key,model=api_model_name)
        else:
            replace_text, replaced_phrases, antonyms,replace_text_w_mark,matches_list,percentage = extract_and_replace_phrases(filter_data[i]['all_output_text'], rate=rate,type='Antonyms',
        base_url=base_url,api_key=api_key,model=api_model_name)
        prompt=replace_text
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs=model.generate(**inputs, **generation_args)
                    #logits = model(**inputs).logits
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            del outputs,inputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            think_match = re.search(r"<think>(.*?)</think>",replace_text_w_mark, re.DOTALL)#提取</think>前的内容
            think_text = think_match.group(0).strip() if think_match else ""
            output_match = re.search(r'</think>\s*(.*)', output_text, re.DOTALL)#提取</think>后的内容
            output_text_wo_think=output_match.group(1).strip() if output_match else ""
            output_w_mark_match = re.search(r'</think>\s*(.*)', replace_text_w_mark, re.DOTALL)#提取</think>后的内容
            output_text_wo_think_w_mark=output_w_mark_match.group(1).strip() if output_w_mark_match else ""
            match = re.findall(r"\\boxed\{([^}]+)\}", output_text,re.DOTALL)
            if len(match)>1:
                model_prediction = get_prediction(match[-1])
            else:
                model_prediction  = None
            correct=1 if model_prediction==filter_data[i]['truth'] else 0
        if synonyms_flag:#change att是指原有被替换的词所占的att(各层各heads取平均)
            record_list.append({'CoT':think_text,'output_text':output_text_wo_think,'output_text_w_mark':output_text_wo_think_w_mark,'prediction':model_prediction,'truth':filter_data[i]['truth'],
                        'correct':correct,'previous_prediction':filter_data[i]['prediction'],'replaced_phrases':replaced_phrases,'synonyms':synonyms,
                        'change_percentage':percentage,'change_attention_percentage_in_CoT':att_data[i]['change_attention_percentage_in_CoT'],'CoT_att_percentage':att_data[i]['CoT_att_percentage'],
                        'input_text_with_CoT':replace_text,'change_attention':att_data[i]['change_attention']})
        else:
            record_list.append({'CoT':think_text,'output_text':output_text_wo_think,'output_text_w_mark':output_text_wo_think_w_mark,'prediction':model_prediction,'truth':filter_data[i]['truth'],
                        'correct':correct,'previous_prediction':filter_data[i]['prediction'],'replaced_phrases':replaced_phrases,'antonyms':antonyms,
                        'change_percentage':percentage,'change_attention_percentage_in_CoT':att_data[i]['change_attention_percentage_in_CoT'],'CoT_att_percentage':att_data[i]['CoT_att_percentage'],
                        'input_text_with_CoT':replace_text,'change_attention':att_data[i]['change_attention']})
    return record_list
#进行topk替换
def topk_replace(model,model_name,filter_data,all_att_list,words_list,synonyms_flag=True,top_k_percentage=0.1):
    base_url,api_key,api_model_name=init_api()
    #获得
    record_list=[]
    for i in tqdm(range(len(filter_data))):
        # if i >0:
        #     break
        replace_words=[]
        if 'Qwen3' in model_name:
            random.seed(51)#51
            model.eval()
            sample=filter_data[i]
            question=sample['question']
            choices=sample['choices'].split('|')
            A_text=choices[0]
            B_text=choices[1]
            C_text=choices[2]
            D_text=choices[3]
            MCQ=MCQ_TEMPLATE.format(q=question,o1=A_text,o2=B_text,o3=C_text,o4=D_text)
            #input_len= len(filter_data[i]['input_text'].split())
            input_len=get_len(filter_data[i]['input_text'])
            # CoT_match = re.search(r"<think>(.*?)</think>",filter_data[i]['all_output_text'], re.DOTALL)#提取</think>前的内容
            # CoT_text = CoT_match.group(0).strip() if CoT_match else ""
            # CoT_match = re.search(r"<think>(.*?)\\boxed",filter_data[i]['all_output_text'], re.DOTALL)#提取答案前的内容
            CoT_match = re.search(r"<think>(.*?)</think>",filter_data[i]['all_output_text'], re.DOTALL)
            CoT_text = CoT_match.group(0) if CoT_match else ""
            explanation_match=re.search(r"</think>(.*?)\\boxed", filter_data[i]['all_output_text'], re.DOTALL) 
            explanation_text=explanation_match.group(1) if explanation_match else ""
            think_len=get_len(CoT_text)+get_len(explanation_text)
            top_k=round(think_len*top_k_percentage)
            topk_indices=get_topk_indices_numpy(all_att_list[i][input_len:(input_len+think_len)],top_k=top_k)# input_len:input_len+CoT_len 代表从CoT开始到CoT结束
            replace_att=[all_att_list[i][j+input_len] for j in topk_indices]#得到topk个att值
            att_sink=all_att_list[i][0]
            value_sum=sum(replace_att)
            percentage_in_all=value_sum/(1-att_sink)
            percentage_in_CoT=value_sum/sum(all_att_list[i][input_len:(input_len+think_len)])
            for j in topk_indices:
                word=words_list[i][j+input_len]
                count=words_list[i][input_len:j+input_len+1].count(word)
                replace_words.append((word,count))
        # else:
        #     topk_indices=get_topk_indices_numpy(all_grad_list[i][input_len:(input_len+CoT_len)],top_k=top_k)
        #     replace_grad=[all_grad_list[i][j+input_len] for j in topk_indices]
        #     value_sum=sum(replace_grad)
        #     percentage_in_all=value_sum/sum(all_grad_list[i])
        #     percentage_in_CoT=value_sum/sum(all_grad_list[i][input_len:(input_len+CoT_len)])
        #     for j in topk_indices:
        #         word=words_list[i][j+input_len]
        #         count=words_list[i][input_len:j+input_len+1].count(word)
        #         replace_words.append((word,count))
        if synonyms_flag:
            replace_text, replaced_phrases, synonyms,replace_text_w_mark,matches_list,percentage = extract_and_replace_phrases(filter_data[i]['all_output_text'],type='Synonyms',
        base_url=base_url,api_key=api_key,model=api_model_name,replace_words=replace_words)
        else:
            replace_text, replaced_phrases, antonyms,replace_text_w_mark,matches_list,percentage = extract_and_replace_phrases(filter_data[i]['all_output_text'],type='Antonyms',
        base_url=base_url,api_key=api_key,model=api_model_name,replace_words=replace_words)
        prompt=replace_text
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs=model.generate(**inputs, **generation_args)
                    #logits = model(**inputs).logits
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            del outputs,inputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            think_match = re.search(r"<think>(.*?)</think>",replace_text_w_mark, re.DOTALL)#提取</think>前的内容
            think_text = think_match.group(0).strip() if think_match else ""
            output_match = re.search(r'</think>\s*(.*)', output_text, re.DOTALL)#提取</think>后的内容
            output_text_wo_think=output_match.group(1).strip() if output_match else ""
            output_w_mark_match = re.search(r'</think>\s*(.*)', replace_text_w_mark, re.DOTALL)#提取</think>后的内容
            output_text_wo_think_w_mark=output_w_mark_match.group(1).strip() if output_w_mark_match else ""
            match = re.findall(r"\\boxed\{([^}]+)\}", output_text,re.DOTALL)
            if len(match)>1:
                model_prediction = get_prediction(match[-1])
            else:
                model_prediction  = None
                #print("synonyms:",model_prediction)
            correct=1 if model_prediction==filter_data[i]['truth'] else 0
        if synonyms_flag:#change att是指原有被替换的词所占的att(各层各heads取平均)
            record_list.append({'CoT':think_text,'output_text':output_text_wo_think,'output_text_w_mark':output_text_wo_think_w_mark,'prediction':model_prediction,'truth':filter_data[i]['truth'],
                        'correct':correct,'previous_prediction':filter_data[i]['prediction'],'replaced_phrases':replaced_phrases,'synonyms':synonyms,
                        'change_percentage':percentage,'input_text_with_CoT':replace_text,'att/grad_in_all':percentage_in_all,
                        'att/grad_in_CoT':percentage_in_CoT,'att/grad_value':value_sum})
        else:
            record_list.append({'CoT':think_text,'output_text':output_text_wo_think,'output_text_w_mark':output_text_wo_think_w_mark,'prediction':model_prediction,'truth':filter_data[i]['truth'],
                        'correct':correct,'previous_prediction':filter_data[i]['prediction'],'replaced_phrases':replaced_phrases,'antonyms':antonyms,
                        'change_percentage':percentage,'input_text_with_CoT':replace_text,'att/grad_in_all':percentage_in_all,
                        'att/grad_in_CoT':percentage_in_CoT,'att/grad_value':value_sum})
    return record_list

if __name__=='__main__':
    root="."
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
    #日志记录
    log_file = f"./log/{model_name}-get_attention_and_gradient-{data_name}_{sub_set}.log"
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
    #data=load_data(data_name,data_path,sub_set)
    os.makedirs(output_path, exist_ok=True)
    tokenizer,model=load_tokenizer_and_model(model_path=model_path)
    model.eval()
    model.gradient_checkpointing_enable()
    logging.info(f"finish load tokenizer and model({model_name})!")
    #模型生成参数
    generation_args = {
        "do_sample": False,
        "max_new_tokens":1024,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        'repetition_penalty':1.2
    }
    #读取数据
    with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_filter_right.json") as f:
        filter_data=json.load(f)
    # with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_prepared_right.json") as f:
    #     prepared_data=json.load(f)#包含input_text和output_text
    logging.info(f"finish load data({data_name}-{sub_set})!")
    #保存获取的attention,gradient和words
    rate=1
    save_list,all_att_list,words_list=get_att(filter_data=filter_data,
    model=model,tokenizer=tokenizer,generation_args=generation_args,rate=rate)
    logging.info("get attention and gradient successful!")
    os.makedirs(f'{root}/results/{data_name}/att_grad_before/', exist_ok=True)
    with open(f'{root}/results/{data_name}/{model_name}_{sub_set}_rate{rate}_right_att.json', 'w', encoding='utf-8') as f:
        json.dump(save_list, f, ensure_ascii=False, indent=4)
    #保存att
    with open(f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_all_att_list.json', 'w', encoding='utf-8') as f:
        json.dump(all_att_list, f, ensure_ascii=False, indent=4)
    #保存grad
    # with open(f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_all_grad_list_correct.json', 'w', encoding='utf-8') as f:
    #     json.dump(all_grad_list_correct, f, ensure_ascii=False, indent=4)
    # with open(f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_all_grad_list_uncorrect.json', 'w', encoding='utf-8') as f:
    #     json.dump(all_grad_list_uncorrect, f, ensure_ascii=False, indent=4)
    #保存words_list
    with open(f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_words_list.json', 'w', encoding='utf-8') as f:
        json.dump(words_list, f, ensure_ascii=False, indent=4)
    # del save_list,all_att_list,words_list
    # torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # with open(f'{root}/results/{data_name}/{model_name}_{sub_set}_rate{rate}_right_att.json') as f:
    #     save_list=json.load(f)
    #CoT随机替换
    output_path=f'{root}/final_results/replace_results/'
    os.makedirs(output_path, exist_ok=True)
    record_list=random_replace(model=model,model_name=model_name,filter_data=filter_data,
                               att_data=save_list,synonyms_flag=True,rate=rate)
    logging.info("finish random replace synonyms!")
    output_file_synonyms=f"{model_name}-{data_name}-{sub_set}-synonym-rate{rate}_random_right.json"
    with open(output_path+output_file_synonyms, 'w', encoding='utf-8') as f:
        json.dump(record_list, f, ensure_ascii=False, indent=4)
    record_list=random_replace(model=model,model_name=model_name,filter_data=filter_data,
                               att_data=save_list,synonyms_flag=False,rate=rate)
    logging.info("finish random replace antonyms!")
    output_file_antonyms=f"{model_name}-{data_name}-{sub_set}-antonym-rate{rate}_random_right.json"
    with open(output_path+output_file_antonyms, 'w', encoding='utf-8') as f:
        json.dump(record_list, f, ensure_ascii=False, indent=4)
    # with open(f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_all_att_list.json') as f:
    #     all_att_list=json.load(f)
    #进行topk替换
    topk_percentage_list=[0.1,0.2,0.3]
    att_flag_list=[True,False]
    synonyms_flag_list=[True,False]
    for synonyms_flag in synonyms_flag_list:
            for top_k_percentage in topk_percentage_list:
                record_list=topk_replace(model=model,model_name=model_name,filter_data=filter_data,all_att_list=all_att_list,words_list=words_list,
                synonyms_flag=synonyms_flag,top_k_percentage=top_k_percentage)
                logging.info(f'finish replace topk! synonyms_flag={synonyms_flag},top_k_percentage={top_k_percentage}')
                #保存结果
                if data_name in ['gpqa','mmlu','mmlu_redux']:
                    output_file_synonyms=f"{model_name}-{data_name}-{sub_set}-synonym-att-right-top_percentage{top_k_percentage}.json"
                    output_file_antonyms=f"{model_name}-{data_name}-{sub_set}-antonym-att-right-top_percentage{top_k_percentage}.json"
                    # else:
                    #     output_file_synonyms=f"{model_name}-{data_name}-{sub_set}-synonym-grad-correct-right-top_percentage{top_k_percentage}.json"#correct代表把正确答案logits最大化然后求导
                    #     output_file_antonyms=f"{model_name}-{data_name}-{sub_set}-antonym-grad-correct-right-top_percentage{top_k_percentage}.json"
                if synonyms_flag:
                    with open(output_path+output_file_synonyms, 'w', encoding='utf-8') as f:
                        json.dump(record_list, f, ensure_ascii=False, indent=4)
                else:
                    with open(output_path+output_file_antonyms, 'w', encoding='utf-8') as f:
                        json.dump(record_list, f, ensure_ascii=False, indent=4)
    logging.info(f"finish get_attention_and_gradient for {data_name}_{sub_set}_{model_name}")
