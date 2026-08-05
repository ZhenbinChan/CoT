from utils import *
import os
import logging
import sys
import json
from tqdm import tqdm
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import string
CONCLUSION_TEMPLATE_1 = '''
{q}
A. {o1}
B. {o2}
C. {o3}
D. {o4}
Here is the thinking draft:
{think}
Please tell me which sentences (no more than five) in the thinking draft you think are the most crucial for getting an answer {answer}, and directly copy them into \\sentence{{}} respectively.
    '''

CONCLUSION_TEMPLATE_K ='''
{q}
A. {o1}
B. {o2}
C. {o3}
D. {o4}
Here is the thinking draft:
{think}
Please tell me which {k} sentence(s) in the thinking draft you think is(are) the most crucial for getting an answer {answer}, and directly copy it(them) into \\sentence{{}} respectively.
    '''

question_TEMPLATE_1 = '''
{q}
A. {o1}
B. {o2}
C. {o3}
D. {o4}
    '''

API_HELP_TEMPLATE='''
This is a conclusion of which sentences are the most crucial for getting an answer {answer}:
{model_output}
Please tell me which sentences in the conclusion are thought as the most crucial, and directly copy them into \\sentence{{}} respectively. 
### Example:
\\sentence{{Here is sentence one.}},
\\sentence{{Here is sentence two.}}
'''
def prepare_data(sample):
    question=sample['question']
    choices_list=sample['choices'].split('|')
    answer=sample['prediction']
    match=re.search(r"<think>(.*?)</think>",sample['all_output_text'],re.DOTALL)
    if match:
        CoT_wo_explanation=match.group(0)
    else:
        CoT_wo_explanation=""
    match=re.search(r'</think>(.*?)\\boxed',sample['all_output_text'],re.DOTALL)
    if match:
        explanation=match.group(1)
    else:
        explanation=""
    CoT_w_explanation=CoT_wo_explanation+explanation
    return question,choices_list,answer,CoT_wo_explanation,CoT_w_explanation,explanation

def get_model_output(model,tokenizer,generation_args,content,think_flag,answer,k):
    if think_flag:
        #think
        prompt=tokenizer.apply_chat_template(
                [{"role":"user","content":content}],
                tokenize=False,#变成ids
                add_generation_prompt=True,#增加assistance
                enable_thinking=True,
            )
        inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs=model.generate(**inputs, **generation_args)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    else:
        #nothink
        prompt=tokenizer.apply_chat_template(
        [{"role":"user","content":content}],
        tokenize=False,#变成ids
        add_generation_prompt=True,#增加assistance
        enable_thinking=False,
        )
        #INPUT_TEMPLATES[model_name].format(user=MCQ,assistant='')
        inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs=model.generate(**inputs, **generation_args)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    match = re.findall(r"\\sentence\{([^}]+)\}",output_text,re.DOTALL)
    if match:
        model_prediction=[]
        for sentence in match:
            if len(model_prediction)>=k:
                break
            model_prediction.append(sentence)
    else:#如果模型没法按格式输出则基于explain部分让外部大模型帮其按格式输出
        match=re.search(r"</think>\s*(.*)",output_text,re.DOTALL)
        if match:
            model_output=match.group(1)
            _,model_prediction = api_help(answer,model_output)
        else:
            match=re.search(r"<think>\s*(.*)",output_text,re.DOTALL)
            if match:
                model_output=match.group(1)
                _,model_prediction = api_help(answer,model_output)
            else:
                model_prediction=None
    return output_text,model_prediction

def api_help(answer,model_output):
    base_url,api_key,api_model_name=init_api()
    query=API_HELP_TEMPLATE.format(answer=answer,model_output=model_output)
    _,output=call_ark(query,api_key, base_url, api_model_name)
    match = re.findall(r"\\sentence\{([^}]+)\}",output,re.DOTALL)
    if match:
        model_prediction=[]
        for sentence in match:
            if len(model_prediction)>=5:
                break
            model_prediction.append(sentence)
    else:
        model_prediction = None
    return output,model_prediction
def find_similar_sentence_bert(embeddings_origin,embeddings_target):
        # 计算相似度
        cosine_scores_list=[]
        for i in range(len(embeddings_origin)):#不包括最后一个target_sentence
            cosine_scores = util.cos_sim(embeddings_target, embeddings_origin[i])
            cosine_scores_list.append(cosine_scores)
        max_index = cosine_scores_list.index(max(cosine_scores_list))
        return max_index

def find_similar_sentence_rouge_L(origin_sentences,target_sentence):
        rouge = Rouge()
        # 计算相似度
        scores_list=[]
        for i in range(len(origin_sentences)):#不包括最后一个target_sentence
            if not origin_sentences[i] or origin_sentences[i].strip() == "" or is_all_punctuation(origin_sentences[i]):
                scores_list.append(0)
            else:
                score = rouge.get_scores(target_sentence, origin_sentences[i])
                scores_list.append(score[0]['rouge-l']['f'])
        max_index = scores_list.index(max(scores_list))
        return max_index

def normalize(s):
    return ''.join(s.split())

def string_match(str1,str2):
    str1_normalize=normalize(str1)
    st2_normalize=normalize(str2)
    if (str1_normalize in st2_normalize) or (st2_normalize in str1_normalize):
        return True
    else:
        return False


def build_kmp_table(pattern):
    table = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and not string_match(pattern[i],pattern[j]):
            j = table[j-1]
        #if pattern[i] == pattern[j]:
        if string_match(pattern[i],pattern[j]):
            j += 1
            table[i] = j
    return table

def kmp_search(words_list, target_words):
    if not target_words:
        return []
    
    table = build_kmp_table(target_words)
    indices = []
    j = 0
    for i in range(len(words_list)):
        while j > 0 and not string_match(words_list[i],target_words[j]):
            j = table[j-1]
        #if words_list[i] == target_words[j]:
        if string_match(words_list[i],target_words[j]):
            j += 1
            if j == len(target_words):
                start = i - j + 1
                indices.extend(range(start, start + j))
                j = table[j-1]  # 继续查找可能的重叠匹配
    return indices

def find_indices_kmp(words_list, sentence):
    target_words = sentence.split()
    return kmp_search(words_list, target_words)

def calculate_metric_1(words_list,sentence,total_att_list,total_att_value):
    index_list = find_indices_kmp(words_list, sentence)
    metric=sum(total_att_list[index_list[0]:index_list[-1]+1])/total_att_value
    return metric

def calculate_metric_2(sentence_list,merge_result_sorted):
    match_result_att=0
    top_att=0
    for i in range(len(sentence_list)):
        match_result_att+=sentence_list[i][1]
        top_att+=merge_result_sorted[i][1]
    metric=match_result_att/top_att
    return metric

def calculate_metric_3(sentence_list,merge_result_sorted,k):
    tp_att=0
    total_att=0
    top_att=0
    for i in range(k):
        top_att+=merge_result_sorted[i][1]
    for i in range(len(sentence_list)):
        total_att+=sentence_list[i][1]
        for j in range(k):
            if string_match(sentence_list[i][0],merge_result_sorted[j][0]):
                tp_att+=sentence_list[i][1]
                break
    precision=tp_att/total_att if total_att!=0 else 0
    recall=tp_att/top_att if top_att!=0 else 0
    if (precision+recall)==0:
        f1=0
    else:
        f1=2*(precision*recall/(precision+recall))
    return precision,recall,f1

def is_all_punctuation(s):
    if not s:
        return False
    
    # 合并中英文标点集合
    en_punctuation = set(string.punctuation)
    cn_punctuation = set("，。！？；：、‘’“”（）【】《》·—…￥")
    all_punctuation = en_punctuation.union(cn_punctuation)
    
    for char in s:
        if char not in all_punctuation:
            return False
    
    return True

if __name__=='__main__':
    root='.'
    data_name,data_path,sub_set,model_name,model_path,output_path=init()
    #日志记录
    log_file = f"./log/{model_name}-metric-{data_name}_{sub_set}.log"
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
    #sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')#,cache_folder='/your/custom/path')
    #sentence_model = SentenceTransformer('/share/nlp/share/plm/paraphrase-multilingual-MiniLM-L12-v2')
    model.eval()
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
    rate=1
    words_list_file=f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_words_list.json'
    all_att_list_file=f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate{rate}_all_att_list.json'
    with open(f"{root}/results/{data_name}/{model_name}_{sub_set}"+"_filter_right.json") as f:
        filter_data=json.load(f)
    with open(words_list_file)as f:
        words_list=json.load(f)
    with open(all_att_list_file) as f:
        all_att_list = json.load(f)
    k_list=[1,2,4,8]
    #for k in k_list:
    record_list=[]
    output_list=[]
    for idx in tqdm(range(len(filter_data))):
        sample=filter_data[idx]
        question,choices_list,answer,CoT_wo_explanation,CoT_w_explanation,explanation=prepare_data(sample)
        thinking_draft=CoT_w_explanation
        template=CONCLUSION_TEMPLATE_1.format(q=question,o1=choices_list[0],o2=choices_list[1],o3=choices_list[2],
                                            o4=choices_list[3],answer=answer,think=thinking_draft)
        question_content=question_TEMPLATE_1.format(q=question,o1=choices_list[0],o2=choices_list[1],o3=choices_list[2],
                                            o4=choices_list[3])
        input_len=get_len(filter_data[idx]['input_text'])
        think_len=get_len(thinking_draft)
        total_att=all_att_list[idx][input_len+1:(input_len+think_len-1)]
        total_words=words_list[idx][input_len+1:(input_len+think_len-1)]
        word_att_tuple=list(zip(total_words,total_att))
        output_text,model_prediction=get_model_output(model,tokenizer,generation_args,content=template,think_flag=True,answer=answer,k=5)
        merge_result=merge_sentences_with_indices_v2(word_att_tuple)
        merge_result_sorted= sorted(merge_result, key=lambda x: x[1], reverse=True)
        origin_sentences = [sentence_tuple[0] for sentence_tuple in merge_result_sorted]
        # bert
        # embeddings_origin = sentence_model.encode(origin_sentences,show_progress_bar=False)
        # similar_sentence_list=[]
        # max_index_list=[]
        # for i in range(len(model_prediction)):
        #     target_sentence=model_prediction[i]
        #     embeddings_target=sentence_model.encode(target_sentence,show_progress_bar=False)
        #     max_index=find_similar_sentence_bert(embeddings_origin,embeddings_target)
        #     if max_index not in max_index_list:
        #         similar_sentence_list.append(merge_result_sorted[max_index])
        #         max_index_list.append(max_index)
        #rouge_L
        similar_sentence_list=[]
        max_index_list=[]
        for i in range(len(model_prediction)):
            target_sentence=model_prediction[i]
            max_index=find_similar_sentence_rouge_L(origin_sentences=origin_sentences,target_sentence=target_sentence)
            if max_index not in max_index_list:
                similar_sentence_list.append(merge_result_sorted[max_index])
                max_index_list.append(max_index)
        #计算找出的句子att占总思考过程的att比例
        #metric_1=calculate_metric_1(words_list=total_words,sentence=similar_sentence,total_att_list=total_att,total_att_value=sum(total_att))
        #计算找出的句子att占最高att句子的att比例
        metric_2=calculate_metric_2(sentence_list=similar_sentence_list,merge_result_sorted=merge_result_sorted)
        precision,recall,f1=calculate_metric_3(sentence_list=similar_sentence_list,merge_result_sorted=merge_result_sorted,k=len(similar_sentence_list))
        #保存数据
        output_list.append({'output_text':output_text})
        record_list.append({'quesion':question_content,'model_answer':sample['prediction'],'truth':sample['truth'],'model_prediction':similar_sentence_list,
                    "top10_sentences":merge_result_sorted[:10],"score":metric_2,"precision":precision,'recall':recall,'f1':f1})
    output_path=f'./final_results/metric/origin/'
    os.makedirs(output_path, exist_ok=True)
    output_file=f'{model_name}-{data_name}-{sub_set}-metric_record(rouge_L).json'
    with open(output_path+output_file, 'w', encoding='utf-8') as f:
        json.dump(record_list, f, ensure_ascii=False, indent=4)
    output_file=f'{model_name}-{data_name}-{sub_set}-metric_model_output.json'
    with open(output_path+output_file, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)
    logging.info(f"finish calculate score for {data_name}_{sub_set}_{model_name}")