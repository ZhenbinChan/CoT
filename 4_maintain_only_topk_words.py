from utils import get_topk_indices_numpy,init,load_tokenizer_and_model,MCQ_TEMPLATE,get_prediction,get_len,maintain_only_topk,get_low_indices_numpy
import json
import re
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import logging
import sys
import random

SENTENCE_END_PATTERN = re.compile(
    r'''[.!?。！？]+(?:["'”’）》」』)\]]*)?(?=\s|$)'''
)


def is_non_terminal_period(text, match):
    """避免把选项标签（如 D.）和英文缩写（如 U.S.）当作句末。"""
    if not match.group(0).startswith('.'):
        return False

    token_match = re.search(r'\S+$', text[:match.end()])
    if token_match is None:
        return False

    token = token_match.group(0).strip('*_([{"\'“‘')
    token = token.rstrip('"\'”’）》」』)]')
    return bool(
        re.fullmatch(r'[A-D]\.', token)
        or re.fullmatch(r'(?:[A-Za-z]\.){2,}', token)
    )


def get_prefix_end_without_last_sentences(text, sentence_count=2):
    """返回删除最后 sentence_count 句话后，保留前缀的字符结束位置。"""
    if sentence_count <= 0 or not text.strip():
        return len(text)

    sentence_spans = []
    sentence_start = 0
    for match in SENTENCE_END_PATTERN.finditer(text):
        if is_non_terminal_period(text, match):
            continue
        sentence_end = match.end()
        if text[sentence_start:sentence_end].strip():
            sentence_spans.append((sentence_start, sentence_end))
        sentence_start = sentence_end

    # 没有结句标点的末尾非空片段也视为一句。
    if text[sentence_start:].strip():
        sentence_spans.append((sentence_start, len(text)))

    if not sentence_spans:
        return len(text)

    remove_count = min(sentence_count, len(sentence_spans))
    return sentence_spans[-remove_count][0]


def get_candidate_indices_after_removing_last_sentences(text, words, sentence_count=2):
    """获得分别删除 CoT 和 explanation 末尾句子后的全局 word 索引。"""
    cot_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    explanation_match = re.search(r"</think>(.*?)\\boxed", text, re.DOTALL)
    word_matches = list(re.finditer(r'\S+', text))

    text_words = [match.group(0) for match in word_matches]
    if text_words != words:
        raise ValueError("words_list 与 all_output_text 的空白分词结果不一致")

    candidate_indices = []
    for segment_match in (cot_match, explanation_match):
        if segment_match is None:
            continue

        segment_text = segment_match.group(1)
        segment_start = segment_match.start(1)
        kept_prefix_end = get_prefix_end_without_last_sentences(
            segment_text,
            sentence_count=sentence_count
        )
        segment_kept_end = segment_start + kept_prefix_end

        candidate_indices.extend(
            index
            for index, word_match in enumerate(word_matches)
            if (
                word_match.start() >= segment_start
                and word_match.end() <= segment_kept_end
            )
        )

    return candidate_indices


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0


def maintain_only_the_rest(rest_words,rest_att_list,top_k):
    random.seed(51)
    #all_rest=' '.join(rest_words)
    for i,word in enumerate(rest_words):
        if ('A' in word) or ('B' in word) or ('C' in word) or ('D' in word):
            del rest_words[i]
    sampled_words,sampled_att_list=maintain_only_random(rest_words,rest_att_list,top_k)
    #sampled_rest=' '.join(sampled_words)
    return rest_words,sampled_words,sampled_att_list

def maintain_only_random(words,att_list,top_k):
    random.seed(51)
    sampled_indices=sorted(random.sample(range(len(words)),top_k))
    sampled_words=[words[i] for i in sampled_indices]
    sampled_att_list=[att_list[i] for i in sampled_indices]
    for i,word in enumerate(sampled_words):
        if ('A' in word) or ('B' in word) or ('C' in word) or ('D' in word):
            del sampled_words[i]
            del sampled_att_list[i]
    return sampled_words,sampled_att_list

def get_maintain_results(replace_words,text,ground_truth,mask_flag,allow_empty=False):
    if allow_empty and not replace_words:
        before_think_match = re.search(r'^(.*?)<think>', text, re.DOTALL)
        replace_text = (
            before_think_match.group(0) + ' </think>'
            if before_think_match else text
        )
    else:
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
#删去直接暴露选项的词
def filter_words(words,indices,input_len,filter_flag,indices_are_global=False):
    del_idx=[]
    replace_words=[]
    for idx,j in enumerate(indices):
        word_index = j if indices_are_global else j + input_len
        word=words[word_index]
        #match=re.search(r'^([A-D][!?.,;:\'"()\-]*)+$',word)#判断word中是否有选项标签，如果有则删除该word
        #if not match:
        if not (('A' in word) or ('B' in word) or ('C' in word) or ('D' in word)):
            count=words[input_len:word_index+1].count(word)
            replace_words.append((word,count))
        else:
            del_idx.append(idx)
    del_idx=sorted(del_idx,reverse=True)
    if filter_flag:
        for idx in del_idx:
            del indices[idx]
    return replace_words

def get_maintain_only_results(model,model_name,filter_data,words_list,all_att_list,top_k_percentage,filter_flag,att_flag,both_flag,remove_last_two_sentences=False):
    mask_flag=False
    record_list=[]
    record_list_all_rest=[]
    record_list_sampled_rest=[]
    record_list_low=[]
    record_list_sampled_all=[]
    for i in tqdm(range(len(filter_data))):
        replace_words=[]
        low_words=[]
        if 'Qwen3' in model_name:
            #random.seed(51)#51
            model.eval()
            sample=filter_data[i]
            question=sample['question']
            choices=sample['choices'].split('|')
            A_text=choices[0]
            B_text=choices[1]
            C_text=choices[2]
            D_text=choices[3]
            MCQ=MCQ_TEMPLATE.format(q=question,o1=A_text,o2=B_text,o3=C_text,o4=D_text)
            # input_len= len(filter_data[i]['input_text'].split())+1#加1保证后续取CoT的时候不包括<think>字符
            # CoT_match = re.search(r"<think>(.*?)\\boxed{",filter_data[i]['all_output_text'], re.DOTALL)#提取</think>前的内容
            # CoT_text = CoT_match.group(1).strip() if CoT_match else ""
            # CoT_len=len(CoT_text.split())#保证后续取CoT的时候不包括</think>字符
            # top_k=round(CoT_len*top_k_percentage)
            input_len=get_len(filter_data[i]['input_text'])
            CoT_match = re.search(r"<think>(.*?)</think>",filter_data[i]['all_output_text'], re.DOTALL)
            CoT_text = CoT_match.group(0) if CoT_match else ""
            explanation_match=re.search(r"</think>(.*?)\\boxed", filter_data[i]['all_output_text'], re.DOTALL) 
            explanation_text=explanation_match.group(1) if explanation_match else ""
            if remove_last_two_sentences:
                candidate_global_indices = get_candidate_indices_after_removing_last_sentences(
                    filter_data[i]['all_output_text'],
                    words_list[i],
                    sentence_count=2
                )
                think_len=len(candidate_global_indices)
                top_k=round(think_len*top_k_percentage)
                think_att_list=[all_att_list[i][j] for j in candidate_global_indices]
                think_words_list=[words_list[i][j] for j in candidate_global_indices]
            else:
                candidate_global_indices = None
                think_len=get_len(CoT_text)+get_len(explanation_text)
                top_k=round(think_len*top_k_percentage)
                think_att_list=all_att_list[i][input_len:(input_len+think_len)]
                think_words_list=words_list[i][input_len:(input_len+think_len)]
        if not both_flag:
            if att_flag:
                topk_indices=get_topk_indices_numpy(think_att_list,top_k=top_k)# input_len:input_len+CoT_len 代表从CoT开始到CoT结束
                low_indices=get_low_indices_numpy(think_att_list,top_k=top_k)
                topk_indices=sorted(topk_indices)
                low_indices=sorted(low_indices)
                mask_id=topk_indices#需要抛弃的部分
                rest_words=[item for j,item in enumerate(think_words_list) if j not in topk_indices]
                rest_att_list=[att for j,att in enumerate(think_att_list) if j not in topk_indices]
                if remove_last_two_sentences:
                    topk_global_indices=[candidate_global_indices[j] for j in topk_indices]
                    low_global_indices=[candidate_global_indices[j] for j in low_indices]
                    replace_words=filter_words(
                        words_list[i],topk_global_indices,input_len,
                        filter_flag=False,indices_are_global=True
                    )
                    low_words=filter_words(
                        words_list[i],low_global_indices,input_len,
                        filter_flag=False,indices_are_global=True
                    )
                    replace_att=[all_att_list[i][j] for j in topk_global_indices]
                    low_att=[all_att_list[i][j] for j in low_global_indices]
                else:
                    replace_words=filter_words(words_list[i],topk_indices,input_len,filter_flag=False)
                    low_words=filter_words(words_list[i],low_indices,input_len,filter_flag=False)
                    replace_att=[all_att_list[i][j+input_len] for j in topk_indices]#得到topk个att值
                    low_att=[all_att_list[i][j+input_len] for j in low_indices]

                att_sink=all_att_list[i][0]
                rest_words,sampled_words,sampled_att_list=maintain_only_the_rest(rest_words,rest_att_list,top_k)
                sampled_all_words,sampled_all_att_list=maintain_only_random(think_words_list,think_att_list,top_k)

                if remove_last_two_sentences:
                    value_sum=sum(replace_att)
                    value_sum_rest=sum(rest_att_list)
                    value_sum_sampled=sum(sampled_att_list)
                    value_sum_low=sum(low_att)
                    value_sum_sampled_all=sum(sampled_all_att_list)

                    candidate_att_sum=sum(think_att_list)
                    all_att_without_sink=1-att_sink
                    percentage_in_all=safe_ratio(value_sum,all_att_without_sink)
                    percentage_in_CoT=safe_ratio(value_sum,candidate_att_sum)
                    percentage_in_all_rest=safe_ratio(value_sum_rest,all_att_without_sink)
                    percentage_in_CoT_rest=safe_ratio(value_sum_rest,candidate_att_sum)
                    percentage_in_all_sampled=safe_ratio(value_sum_sampled,all_att_without_sink)
                    percentage_in_CoT_sampled=safe_ratio(value_sum_sampled,candidate_att_sum)
                    percentage_in_all_low=safe_ratio(value_sum_low,all_att_without_sink)
                    percentage_in_CoT_low=safe_ratio(value_sum_low,candidate_att_sum)
                    percentage_in_all_sampled_all=safe_ratio(value_sum_sampled_all,all_att_without_sink)
                    percentage_in_CoT_sampled_all=safe_ratio(value_sum_sampled_all,candidate_att_sum)
                else:
                    value_sum=sum(replace_att) if 0 not in topk_indices else sum(replace_att)-att_sink
                    value_sum_rest=sum(rest_att_list)-att_sink if 0 not in topk_indices else sum(rest_att_list)
                    value_sum_sampled=sum(sampled_att_list)-att_sink if words_list[i][0] in sampled_words else sum(sampled_att_list)
                    value_sum_low=sum(low_att)
                    value_sum_sampled_all=sum(sampled_all_att_list)

                    percentage_in_all=value_sum/(1-att_sink)
                    percentage_in_CoT=value_sum/sum(think_att_list)
                    percentage_in_all_rest=value_sum_rest/(1-att_sink)
                    percentage_in_CoT_rest=value_sum_rest/sum(think_att_list)
                    percentage_in_all_sampled=value_sum_sampled/(1-att_sink)
                    percentage_in_CoT_sampled=value_sum_sampled/sum(think_att_list)
                    percentage_in_all_low=value_sum_low/(1-att_sink)
                    percentage_in_CoT_low=value_sum_low/sum(think_att_list)
                    percentage_in_all_sampled_all=value_sum_sampled_all/(1-att_sink)
                    percentage_in_CoT_sampled_all=value_sum_sampled_all/sum(think_att_list)
                #replace_words=[words_list[i][j+input_len] for j in topk_indices] 
            # else:
            #     topk_indices=get_topk_indices_numpy(all_grad_list[i][input_len:(input_len+CoT_len)],top_k=top_k)
            #     topk_indices=sorted(topk_indices)
            #     del_idx=[]
            #     for idx,j in enumerate(topk_indices):
            #         word=words_list[i][j+input_len]
            #         match=re.search(r'^([A-D][!?.,;:\'"()\-]*)+$',word)#判断word中是否有选项标签，如果有则删除该word
            #         if not match:
            #             count=words_list[i][input_len:j+input_len+1].count(word)
            #             replace_words.append((word,count))
            #         else:
            #             del_idx.append(idx)
            #     del_idx=sorted(del_idx,reverse=True)
            #     if filter_flag:
            #         for idx in del_idx:
            #             del topk_indices[idx]
            #     replace_grad=[all_grad_list[i][j+input_len] for j in topk_indices]
            #     #replace_words=[words_list[i][j+input_len] for j in topk_indices] 
            #     value_sum=sum(replace_grad)
            #     percentage_in_all=value_sum/sum(all_grad_list[i])
            #     percentage_in_CoT=value_sum/sum(all_grad_list[i][input_len:(input_len+CoT_len)])
        # else:
        #     topk_indices_att=get_topk_indices_numpy(all_att_list[i][input_len:(input_len+CoT_len)],top_k=top_k)
        #     topk_indices_grad=get_topk_indices_numpy(all_grad_list[i][input_len:(input_len+CoT_len)],top_k=top_k)
        #     topk_indices= list(set(topk_indices_att) & set(topk_indices_grad))#取交集
        #     topk_indices=sorted(topk_indices)
        #     del_idx=[]
        #     for idx,j in enumerate(topk_indices):
        #         word=words_list[i][j+input_len]
        #         match=re.search(r'^([A-D][!?.,;:\'"()\-]*)+$',word)#判断word中是否有选项标签，如果有则删除该word
        #         if not match:
        #             count=words_list[i][input_len:j+input_len+1].count(word)
        #             replace_words.append((word,count))
        #         else:
        #             del_idx.append(idx)
        #     del_idx=sorted(del_idx,reverse=True)
        #     if filter_flag:
        #         for idx in del_idx:
        #             del topk_indices[idx]
        #     replace_att=[all_att_list[i][j+input_len] for j in topk_indices]#得到topk个att值
        #     att_sink=all_att_list[i][0]
        #     value_sum=sum(replace_att)
        #     percentage_in_all=value_sum/(1-att_sink)
        #     percentage_in_CoT=value_sum/sum(all_att_list[i][input_len:(input_len+CoT_len)])
        mode_metadata = (
            {'remove_last_two_sentences': True}
            if remove_last_two_sentences else {}
        )
        # 只保留topk keywords       
        think_text,output_text_wo_think,model_prediction,correct,replace_text=get_maintain_results(
            replace_words,filter_data[i]['all_output_text'],filter_data[i]['truth'],mask_flag,
            allow_empty=remove_last_two_sentences
        )
        record_list.append({'sample_index':i,'question':filter_data[i]['question'],
                    **mode_metadata,
                    'original_CoT':filter_data[i]['CoT'],
                    'original_explanation':filter_data[i]['output_text'],
                    'original_answer':filter_data[i]['prediction'],'modified_CoT':think_text,
                    'CoT':think_text,'output_text':output_text_wo_think,'prediction':model_prediction,'truth':filter_data[i]['truth'],
                    'correct':correct,'previous_prediction':filter_data[i]['prediction'],
                    'input_text_with_CoT':replace_text,'att/grad_in_all':percentage_in_all,
                    'att/grad_in_CoT':percentage_in_CoT,'att/grad_value':value_sum})
        #保留全部rest
        think_text,output_text_wo_think,model_prediction,correct,replace_text=get_maintain_results(
            rest_words,filter_data[i]['all_output_text'],filter_data[i]['truth'],mask_flag,
            allow_empty=remove_last_two_sentences
        )
        record_list_all_rest.append({'sample_index':i,'question':filter_data[i]['question'],
                    **mode_metadata,
                    'original_CoT':filter_data[i]['CoT'],
                    'original_explanation':filter_data[i]['output_text'],
                    'original_answer':filter_data[i]['prediction'],'modified_CoT':think_text,
                    'CoT':think_text,'output_text':output_text_wo_think,'prediction':model_prediction,'truth':filter_data[i]['truth'],
                    'correct':correct,'previous_prediction':filter_data[i]['prediction'],
                    'input_text_with_CoT':replace_text,'att/grad_in_all':percentage_in_all_rest,
                    'att/grad_in_CoT':percentage_in_CoT_rest,'att/grad_value':value_sum_rest})
        #保留sampled rest
        think_text,output_text_wo_think,model_prediction,correct,replace_text=get_maintain_results(
            sampled_words,filter_data[i]['all_output_text'],filter_data[i]['truth'],mask_flag,
            allow_empty=remove_last_two_sentences
        )
        record_list_sampled_rest.append({'sample_index':i,'question':filter_data[i]['question'],
                    **mode_metadata,
                    'original_CoT':filter_data[i]['CoT'],
                    'original_explanation':filter_data[i]['output_text'],
                    'original_answer':filter_data[i]['prediction'],'modified_CoT':think_text,
                    'CoT':think_text,'output_text':output_text_wo_think,'prediction':model_prediction,'truth':filter_data[i]['truth'],
                    'correct':correct,'previous_prediction':filter_data[i]['prediction'],
                    'input_text_with_CoT':replace_text,'att/grad_in_all':percentage_in_all_sampled,
                    'att/grad_in_CoT':percentage_in_CoT_sampled,'att/grad_value':value_sum_sampled})
        #保留low attention words
        think_text,output_text_wo_think,model_prediction,correct,replace_text=get_maintain_results(
            low_words,filter_data[i]['all_output_text'],filter_data[i]['truth'],mask_flag,
            allow_empty=remove_last_two_sentences
        )
        record_list_low.append({'sample_index':i,'question':filter_data[i]['question'],
                    **mode_metadata,
                    'original_CoT':filter_data[i]['CoT'],
                    'original_explanation':filter_data[i]['output_text'],
                    'original_answer':filter_data[i]['prediction'],'modified_CoT':think_text,
                    'CoT':think_text,'output_text':output_text_wo_think,'prediction':model_prediction,'truth':filter_data[i]['truth'],
                    'correct':correct,'previous_prediction':filter_data[i]['prediction'],
                    'input_text_with_CoT':replace_text,'att/grad_in_all':percentage_in_all_low,
                    'att/grad_in_CoT':percentage_in_CoT_low,'att/grad_value':value_sum_low})
        #保留sampled all words
        think_text,output_text_wo_think,model_prediction,correct,replace_text=get_maintain_results(
            sampled_all_words,filter_data[i]['all_output_text'],filter_data[i]['truth'],mask_flag,
            allow_empty=remove_last_two_sentences
        )
        record_list_sampled_all.append({'sample_index':i,'question':filter_data[i]['question'],
                    **mode_metadata,
                    'original_CoT':filter_data[i]['CoT'],
                    'original_explanation':filter_data[i]['output_text'],
                    'original_answer':filter_data[i]['prediction'],'modified_CoT':think_text,
                    'CoT':think_text,'output_text':output_text_wo_think,'prediction':model_prediction,'truth':filter_data[i]['truth'],
                    'correct':correct,'previous_prediction':filter_data[i]['prediction'],
                    'input_text_with_CoT':replace_text,'att/grad_in_all':percentage_in_all_sampled_all,
                    'att/grad_in_CoT':percentage_in_CoT_sampled_all,'att/grad_value':value_sum_sampled_all})
    return record_list,record_list_all_rest,record_list_sampled_rest,record_list_low,record_list_sampled_all


if __name__=='__main__':
    root='.'
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
      #日志记录
    log_file = f"./log/{model_name}-maintain_only_topk-{data_name}_{sub_set}.log"
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
    #with open(f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_all_grad_list_correct.json', encoding='utf-8') as f:
        #all_grad_list = json.load(f)
    top_k_percentage_list=[0.1,0.2,0.3] 
    att_flag_list=[True]
    both_flag_list=[False]
    filter_flag=True
    REMOVE_LAST_TWO_SENTENCES=False
    if REMOVE_LAST_TWO_SENTENCES:
        logging.warning(
            "REMOVE_LAST_TWO_SENTENCES=True: 将覆盖原路径中的同名实验结果。"
        )
    for both_flag in both_flag_list:
        for att_flag in att_flag_list:  
            for top_k_percentage in top_k_percentage_list:  
                record_list,record_list_all_rest,record_list_sampled_rest,record_list_low,record_list_sampled_all=get_maintain_only_results(
                    model,model_name,filter_data,words_list,all_att_list,top_k_percentage,
                    filter_flag=filter_flag,att_flag=att_flag,both_flag=both_flag,
                    remove_last_two_sentences=REMOVE_LAST_TWO_SENTENCES
                )
                logging.info("get maintain only results successfully!")
                if filter_flag:
                    output_path=f'{root}/final_results/maintain_only_topk/filter/'
                    output_path_rest=f'{root}/final_results/maintain_rest/filter/'
                    output_path_sampled=f'{root}/final_results/maintain_sampled/filter/'
                    output_path_low=f'{root}/final_results/maintain_low/filter/'
                    output_path_sampled_all=f'{root}/final_results/maintain_sampled_all/filter/'
                else:
                    output_path=f'{root}/final_results/maintain_only_topk/'
                    output_path_rest=f'{root}/final_results/maintain_rest/'
                    output_path_sampled=f'{root}/final_results/maintain_sampled/'
                    output_path_low=f'{root}/final_results/maintain_low/'
                    output_path_sampled_all=f'{root}/final_results/maintain_sampled_all/'
                os.makedirs(output_path, exist_ok=True)
                os.makedirs(output_path_rest, exist_ok=True)
                os.makedirs(output_path_sampled, exist_ok=True)
                os.makedirs(output_path_low, exist_ok=True)
                os.makedirs(output_path_sampled_all, exist_ok=True)
                # if mask_flag:
                #     mask_path='mask/'
                # else:
                mask_path='unmask/'
                if not both_flag:
                    if att_flag:
                        output_file=f"{model_name}-{data_name}-{sub_set}-att-right-top_percentage{top_k_percentage}.json"
                    else:
                        output_file=f"{model_name}-{data_name}-{sub_set}-grad-correct-right-top_percentage{top_k_percentage}.json"
                else:
                    output_file=f"{model_name}-{data_name}-{sub_set}-both-grad_correct-right-top_percentage{top_k_percentage}.json"
                os.makedirs(output_path+mask_path, exist_ok=True)
                os.makedirs(output_path_rest+mask_path, exist_ok=True)
                os.makedirs(output_path_sampled+mask_path, exist_ok=True)
                os.makedirs(output_path_low+mask_path, exist_ok=True)
                os.makedirs(output_path_sampled_all+mask_path, exist_ok=True)
                with open(output_path+mask_path+output_file, 'w', encoding='utf-8') as f:
                    json.dump(record_list, f, ensure_ascii=False, indent=4)
                with open(output_path_rest+mask_path+output_file, 'w', encoding='utf-8') as f:
                    json.dump(record_list_all_rest, f, ensure_ascii=False, indent=4)
                with open(output_path_sampled+mask_path+output_file, 'w', encoding='utf-8') as f:
                    json.dump(record_list_sampled_rest, f, ensure_ascii=False, indent=4)
                with open(output_path_low+mask_path+output_file, 'w', encoding='utf-8') as f:
                    json.dump(record_list_low, f, ensure_ascii=False, indent=4)
                with open(output_path_sampled_all+mask_path+output_file, 'w', encoding='utf-8') as f:
                    json.dump(record_list_sampled_all, f, ensure_ascii=False, indent=4)
                logging.info(f"save maintain only results successfully! both_flag={both_flag},att_flag={att_flag},top_k_percentage={top_k_percentage}")
    logging.info(f"finish maintain only for {data_name}_{sub_set}_{model_name}")
