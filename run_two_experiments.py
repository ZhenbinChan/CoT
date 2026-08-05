"""
合并实验脚本：
实验A - 只保留高Attention句子，看模型答题效果
实验B - 只保留模型自己认为重要的句子，看模型答题效果

运行方式：
    python3 run_two_experiments.py

依赖前置文件：
    - results/att_IG_words/{data_name}/{model_name}/ATT_list_all.json
    - results/att_IG_words/{data_name}/{model_name}/words_list_all.json
    - results/model_output/{data_name}/{model_name}/model_output_think_data_all_filter.json
"""

import json
import os
import re
import random
import logging
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from rouge import Rouge
import string
from utils import (
    init, load_tokenizer_and_model, get_prediction,
    merge_sentences_with_indices_v2, get_topk_indices_numpy,
    get_len, call_ark, init_api
)

def save_record(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ============================================================
# Prompt 模板
# ============================================================
CONCLUSION_TEMPLATE = '''
{q}
A. {o1}
B. {o2}
C. {o3}
D. {o4}
Here is the thinking draft:
{think}
Please tell me which sentences (no more than five) in the thinking draft you think are the most crucial for getting an answer {answer}, and directly copy them into \\sentence{{}} respectively.
'''

API_HELP_TEMPLATE = '''
This is a conclusion of which sentences are the most crucial for getting an answer {answer}:
{model_output}
Please tell me which sentences in the conclusion are thought as the most crucial, and directly copy them into \\sentence{{}} respectively.
### Example:
\\sentence{{Here is sentence one.}},
\\sentence{{Here is sentence two.}}
'''

# ============================================================
# 工具函数
# ============================================================
def is_all_punctuation(s):
    """判断字符串是否全为标点"""
    if not s:
        return False
    en_punctuation = set(string.punctuation)
    cn_punctuation = set("，。！？；：、''""（）【】《》·—…￥")
    all_punctuation = en_punctuation.union(cn_punctuation)
    return all(char in all_punctuation for char in s)

def construct_CoT(sentence_list=None):
    """把句子列表拼成CoT格式"""
    CoT = ''
    if sentence_list:
        for sentence in sentence_list:
            CoT += sentence + ' '
    final_CoT = '<think>' + CoT.strip() + '</think>\n\nThe correct answer is \\boxed{'
    return final_CoT

def get_logits(model, tokenizer, text):
    """获取模型最后一个token的logits"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs_logits = model(**inputs).logits[:, -1, :]
    return outputs_logits

def get_answer_from_logits(logits, answer_ids):
    """根据logits和answer_ids得到预测答案"""
    answer_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    probs = F.softmax(logits.squeeze().detach().cpu(), dim=-1)
    answer_probs = [probs[idx].item() for idx in answer_ids]
    return answer_dict[answer_probs.index(max(answer_probs))]

def find_similar_sentence_rouge_L(origin_sentences, target_sentence):
    """用Rouge-L找最相似的句子"""
    rouge = Rouge()
    scores_list = []
    for s in origin_sentences:
        if not s or s.strip() == "" or is_all_punctuation(s):
            scores_list.append(0)
        else:
            try:
                score = rouge.get_scores(target_sentence, s)
                scores_list.append(score[0]['rouge-l']['f'])
            except:
                scores_list.append(0)
    return scores_list.index(max(scores_list))

def api_help(answer, model_output):
    """调用外部API辅助提取句子"""
    base_url, api_key, api_model_name = init_api()
    query = API_HELP_TEMPLATE.format(answer=answer, model_output=model_output)
    _, output = call_ark(query, api_key, base_url, api_model_name)
    match = re.findall(r"\\sentence\{([^}]+)\}", output, re.DOTALL)
    if match:
        return [s for s in match[:5]]
    return None

# ============================================================
# 步骤1：把token-level attention聚合成sentence-level
# ============================================================
def build_sentence_level_att(output_data, value_data, words_data):
    """
    输入：
        output_data  - 模型输出数据列表
        value_data   - token级别attention列表
        words_data   - token对应的词列表
    输出：
        merge_result_list - 每条数据的[(句子, att值), ...]列表
    """
    merge_result_list = []
    for idx in tqdm(range(len(output_data)), desc="聚合sentence-level attention"):
        input_len = get_len(output_data[idx]['input_text'])
        CoT_match = re.search(r"<think>(.*?)</think>", output_data[idx]['all_output_text'], re.DOTALL)
        CoT_text = CoT_match.group(0) if CoT_match else ""
        CoT_len = get_len(CoT_text)

        if value_data[idx]:
            total_value = value_data[idx][input_len + 1:(input_len + CoT_len - 1)]
            total_words = words_data[idx][input_len + 1:(input_len + CoT_len - 1)]
            word_value_tuple = list(zip(total_words, total_value))
            merge_result = merge_sentences_with_indices_v2(word_value_tuple)
            merge_result_list.append(merge_result)
        else:
            merge_result_list.append(None)

    return merge_result_list

# ============================================================
# 实验A：只保留高Attention句子，用logits判断答题效果
# ============================================================
def run_experiment_A(output_data, merge_result_list, model, tokenizer, answer_ids, top_k_percentage_list):
    """
    对每条数据：
      1. 选出topk高attention句子
      2. 构造新CoT喂给模型
      3. 对比答案是否正确
    """
    results = []
    answer_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for i in tqdm(range(len(output_data)), desc="实验A-高Attention句子"):
        if merge_result_list[i] is None:
            results.append(None)
            continue

        item = {}
        input_text = output_data[i]['input_text']
        truth = output_data[i]['truth']
        original_prediction = output_data[i]['prediction']
        merge_result = merge_result_list[i]
        sentence_list = [x[0] for x in merge_result]
        sentence_att_list = [x[1] for x in merge_result]

        # nothink基线
        nothink_CoT = construct_CoT()
        nothink_text = input_text + nothink_CoT
        nothink_logits = get_logits(model, tokenizer, nothink_text)
        nothink_answer = get_answer_from_logits(nothink_logits, answer_ids)

        # think完整CoT
        think_CoT = output_data[i]['CoT'] + '</think>\n\nThe correct answer is \\boxed{'
        think_text = input_text + think_CoT
        think_logits = get_logits(model, tokenizer, think_text)
        think_answer = get_answer_from_logits(think_logits, answer_ids)

        item['truth'] = truth
        item['original_prediction'] = original_prediction
        item['nothink_answer'] = nothink_answer
        item['think_answer'] = think_answer
        item['nothink_correct'] = 1 if nothink_answer == truth else 0
        item['think_correct'] = 1 if think_answer == truth else 0

        for top_k_percentage in top_k_percentage_list:
            top_k = max(1, round(len(merge_result) * top_k_percentage))

            # 高attention句子
            topk_indices = sorted(get_topk_indices_numpy(sentence_att_list, top_k=top_k))
            maintain_top_sentences = [sentence_list[j] for j in topk_indices]
            maintain_top_CoT = construct_CoT(maintain_top_sentences)
            maintain_top_text = input_text + maintain_top_CoT
            maintain_top_logits = get_logits(model, tokenizer, maintain_top_text)
            maintain_top_answer = get_answer_from_logits(maintain_top_logits, answer_ids)

            # 低attention句子（对照组）
            low_indices = sorted(get_topk_indices_numpy([-v for v in sentence_att_list], top_k=top_k))
            maintain_low_sentences = [sentence_list[j] for j in low_indices]
            maintain_low_CoT = construct_CoT(maintain_low_sentences)
            maintain_low_text = input_text + maintain_low_CoT
            maintain_low_logits = get_logits(model, tokenizer, maintain_low_text)
            maintain_low_answer = get_answer_from_logits(maintain_low_logits, answer_ids)

            # 随机句子（对照组）
            random.seed(42)
            random_indices = sorted(random.sample(range(len(sentence_list)), top_k))
            maintain_random_sentences = [sentence_list[j] for j in random_indices]
            maintain_random_CoT = construct_CoT(maintain_random_sentences)
            maintain_random_text = input_text + maintain_random_CoT
            maintain_random_logits = get_logits(model, tokenizer, maintain_random_text)
            maintain_random_answer = get_answer_from_logits(maintain_random_logits, answer_ids)

            item[f'top{top_k_percentage}_answer'] = maintain_top_answer
            item[f'top{top_k_percentage}_correct'] = 1 if maintain_top_answer == truth else 0
            item[f'low{top_k_percentage}_answer'] = maintain_low_answer
            item[f'low{top_k_percentage}_correct'] = 1 if maintain_low_answer == truth else 0
            item[f'random{top_k_percentage}_answer'] = maintain_random_answer
            item[f'random{top_k_percentage}_correct'] = 1 if maintain_random_answer == truth else 0

        results.append(item)

    return results

# ============================================================
# 实验B：让模型说出重要句子，再验证答题效果
# ============================================================
def ask_model_important_sentences(output_data, model, tokenizer, generation_args):
    """
    用CONCLUSION_TEMPLATE让模型说出它认为最重要的句子
    返回每条数据模型认为重要的句子列表
    """
    model_selected_list = []

    for i in tqdm(range(len(output_data)), desc="实验B-让模型选句子"):
        sample = output_data[i]
        # 从all_output_text里提取question和choices
        input_match = re.search(r'^(.*?)<think>', sample['all_output_text'], re.DOTALL)
        question_block = input_match.group(1).strip() if input_match else sample['input_text']

        # 提取CoT
        CoT_match = re.search(r"<think>(.*?)</think>", sample['all_output_text'], re.DOTALL)
        CoT_text = CoT_match.group(1).strip() if CoT_match else ""

        answer = sample['prediction'] if sample['prediction'] else 'A'

        # 构造prompt（简化版，直接把question_block和CoT放进去）
        content = f"{question_block}\n\nHere is the thinking draft:\n{CoT_text}\n\nPlease tell me which sentences (no more than five) in the thinking draft you think are the most crucial for getting an answer {answer}, and directly copy them into \\sentence{{}} respectively."

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_args)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 提取\sentence{}
        match = re.findall(r"\\sentence\{([^}]+)\}", output_text, re.DOTALL)
        if match:
            selected_sentences = match[:5]
        else:
            # 尝试用API辅助提取
            try:
                selected_sentences = api_help(answer, output_text)
            except:
                selected_sentences = None

        model_selected_list.append({
            'selected_sentences': selected_sentences,
            'raw_output': output_text
        })

    return model_selected_list

def run_experiment_B(output_data, model_selected_list, merge_result_list, model, tokenizer, answer_ids):
    """
    对每条数据：
      1. 用Rouge-L把模型选的句子映射到merge_result里的句子
      2. 构造新CoT喂给模型
      3. 对比答案是否正确
    """
    results = []

    for i in tqdm(range(len(output_data)), desc="实验B-验证模型选句效果"):
        if merge_result_list[i] is None or model_selected_list[i]['selected_sentences'] is None:
            results.append(None)
            continue

        item = {}
        input_text = output_data[i]['input_text']
        truth = output_data[i]['truth']
        merge_result = merge_result_list[i]
        origin_sentences = [x[0] for x in merge_result]
        selected_sentences = model_selected_list[i]['selected_sentences']

        # 用Rouge-L找最相似的句子
        similar_sentence_list = []
        max_index_list = []
        for target_sentence in selected_sentences:
            max_index = find_similar_sentence_rouge_L(origin_sentences, target_sentence)
            if max_index not in max_index_list:
                similar_sentence_list.append(merge_result[max_index][0])
                max_index_list.append(max_index)

        # 构造新CoT
        selected_CoT = construct_CoT(similar_sentence_list)
        selected_text = input_text + selected_CoT
        selected_logits = get_logits(model, tokenizer, selected_text)
        selected_answer = get_answer_from_logits(selected_logits, answer_ids)

        # 计算attention得分（模型选的句子att占总att的比例）
        sentence_att_list = [x[1] for x in merge_result]
        total_att = sum(sentence_att_list)
        selected_att = sum([merge_result[j][1] for j in max_index_list])
        att_ratio = selected_att / total_att if total_att > 0 else 0

        item['truth'] = truth
        item['original_prediction'] = output_data[i]['prediction']
        item['selected_sentences'] = similar_sentence_list
        item['selected_answer'] = selected_answer
        item['selected_correct'] = 1 if selected_answer == truth else 0
        item['att_ratio'] = att_ratio  # 模型选的句子占总att的比例

        results.append(item)

    return results

# ============================================================
# 统计并打印结果
# ============================================================
def print_summary(results_A, results_B, top_k_percentage_list):
    print("\n" + "="*60)
    print("实验A结果：保留高Attention句子的答题正确率")
    print("="*60)

    valid = [r for r in results_A if r is not None]
    total = len(valid)
    if total == 0:
        print("没有有效数据！")
    else:
        nothink_acc = sum(r['nothink_correct'] for r in valid) / total * 100
        think_acc = sum(r['think_correct'] for r in valid) / total * 100
        print(f"{'条件':<20} {'正确率':>10}")
        print(f"{'nothink (无CoT)':<20} {nothink_acc:>9.1f}%")
        print(f"{'think (完整CoT)':<20} {think_acc:>9.1f}%")
        for p in top_k_percentage_list:
            top_acc = sum(r[f'top{p}_correct'] for r in valid) / total * 100
            low_acc = sum(r[f'low{p}_correct'] for r in valid) / total * 100
            rnd_acc = sum(r[f'random{p}_correct'] for r in valid) / total * 100
            print(f"{'高ATT top'+str(int(p*100))+'%':<20} {top_acc:>9.1f}%")
            print(f"{'低ATT low'+str(int(p*100))+'%':<20} {low_acc:>9.1f}%")
            print(f"{'随机random'+str(int(p*100))+'%':<20} {rnd_acc:>9.1f}%")

    print("\n" + "="*60)
    print("实验B结果：保留模型自选句子的答题正确率")
    print("="*60)

    valid_B = [r for r in results_B if r is not None]
    total_B = len(valid_B)
    if total_B == 0:
        print("没有有效数据！")
    else:
        selected_acc = sum(r['selected_correct'] for r in valid_B) / total_B * 100
        avg_att_ratio = sum(r['att_ratio'] for r in valid_B) / total_B * 100
        print(f"{'模型自选句子正确率':<20} {selected_acc:>9.1f}%")
        print(f"{'模型选句的平均ATT占比':<20} {avg_att_ratio:>9.1f}%")
    print("="*60)

# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    root = "."
    data_name, data_path, sub_set, model_name, model_path, output_path = init()

    # 日志设置
    log_file = f"./log/{model_name}-two_experiments-{data_name}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # ---------- 加载数据 ----------
    logging.info("加载数据...")
    att_path = f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate1_all_att_list.json'
    words_path = f'{root}/results/{data_name}/att_grad_before/{model_name}_{sub_set}_rate1_words_list.json'
    output_data_path = f'{root}/results/{data_name}/{model_name}_{sub_set}_filter_right.json'

    with open(att_path) as f:
        value_data = json.load(f)
    with open(words_path) as f:
        words_data = json.load(f)
    with open(output_data_path) as f:
        output_data = json.load(f)

    logging.info(f"数据条数: {len(output_data)}")

    # ---------- 加载模型 ----------
    logging.info("加载模型...")
    tokenizer, model = load_tokenizer_and_model(model_path)
    model.eval()
    logging.info(f"模型加载完成: {model_name}")

    # 获取ABCD对应的token id
    answer_list = ['A', 'B', 'C', 'D']
    answer_ids = [tokenizer(a).input_ids[0] for a in answer_list]

    # 生成参数
    generation_args_select = {
        "do_sample": False,
        "max_new_tokens": 2048,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        "repetition_penalty": 1.2
    }

    top_k_percentage_list = [0.1, 0.2, 0.3]

    # ---------- 聚合sentence-level attention ----------
    logging.info("聚合sentence-level attention...")
    merge_result_list = build_sentence_level_att(output_data, value_data, words_data)

    # 保存中间结果
    os.makedirs(f'{root}/results/two_experiments/{data_name}/{model_name}/{sub_set}/', exist_ok=True)
    save_record(
        [[list(t) for t in r] if r else None for r in merge_result_list],
        f'{root}/results/two_experiments/{data_name}/{model_name}/{sub_set}/merge_result.json'
    )
    logging.info("sentence-level attention聚合完成！")

    # ---------- 实验A ----------
    logging.info("开始实验A：高Attention句子...")
    results_A = run_experiment_A(
        output_data, merge_result_list, model, tokenizer,
        answer_ids, top_k_percentage_list
    )
    save_record(
        results_A,
        f'{root}/results/two_experiments/{data_name}/{model_name}/{sub_set}/experiment_A_results.json'
    )
    logging.info("实验A完成！")

    # ---------- 实验B ----------
    logging.info("开始实验B：让模型选重要句子...")
    model_selected_list = ask_model_important_sentences(
        output_data, model, tokenizer, generation_args_select
    )
    save_record(
        model_selected_list,
        f'{root}/results/two_experiments/{data_name}/{model_name}/{sub_set}/model_selected_sentences.json'
    )

    results_B = run_experiment_B(
        output_data, model_selected_list, merge_result_list,
        model, tokenizer, answer_ids
    )
    save_record(
        results_B,
        f'{root}/results/two_experiments/{data_name}/{model_name}/{sub_set}/experiment_B_results.json'
    )
    logging.info("实验B完成！")

    # ---------- 打印汇总结果 ----------
    print_summary(results_A, results_B, top_k_percentage_list)
    logging.info(f"全部实验完成！结果保存在 ./results/two_experiments/{data_name}/{model_name}/{sub_set}/")
