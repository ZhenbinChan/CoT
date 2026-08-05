try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from liger_kernel.transformers import AutoLigerKernelForCausalLM
#from modeling.modeling_qwen3_modified import Qwen3ForCausalLM
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import gc
import re
from torch.utils.checkpoint import checkpoint
from transformers import LogitsProcessor
#初始化数据集等参数


def init():

    parser = argparse.ArgumentParser()

    # =====================================================
    # index 参数
    # =====================================================

    parser.add_argument(
        "--data_idx",
        type=int,
        default=4
    )

    parser.add_argument(
        "--subset_idx",
        type=int,
        default=2
    )

    parser.add_argument(
        "--model_idx",
        type=int,
        default=4
    )

    # 允许直接指定名字（优先级更高）
    parser.add_argument(
        "--data_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--sub_set",
        type=str,
        default=None
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None
    )

    args = parser.parse_args()

    # =====================================================
    # data
    # =====================================================

    data_name_dict = {
        0: 'gpqa',
        1: 'mmlu',
        2: 'ComVE',
        3: 'esnli',
        4: 'mmlu_redux',
    }

    data_path_dict = {
        'gpqa': 'Idavidrein/gpqa',
        'mmlu': 'cais/mmlu',
        'ComVE': './datasets/ComVE/dev.csv',
        'esnli': './datasets/eSNLI/esnli_dev.csv',
        'mmlu_redux': 'edinburgh-dawg/mmlu-redux-2.0'
    }

    # =====================================================
    # subset
    # =====================================================

    subset_dict = {
        0: 'global_facts',
        1: 'high_school_mathematics',
        2: 'college_mathematics',
        3: 'high_school_computer_science',
        4: 'college_computer_science',
        5: 'high_school_biology',
        6: 'college_biology',
        7: 'professional_law',
        8: 'college_physics',
        9: 'machine_learning',
        10: 'sociology',
        11: 'us_foreign_policy',
        12: 'computer_security',
        13: 'conceptual_physics',
        14: 'econometrics',
        15: 'business_ethics',
        16: 'clinical_knowledge',
        17: 'electrical_engineering',
        18: 'elementary_mathematics',
        19: 'formal_logic'
    }

    # =====================================================
    # model
    # =====================================================

    model_dict = {
        0: 'DeepSeek-R1-Distill-Qwen-7B',
        1: 'DeepSeek-R1-Distill-Llama-8B',
        2: 'Qwen3-8B',
        3: 'Qwen3-4B',
        4: 'Qwen3-0.6B'
    }

    # =====================================================
    # 优先使用命令行参数
    # =====================================================

    if args.data_name is not None:
        data_name = args.data_name
    else:
        data_name = data_name_dict[args.data_idx]

    if args.sub_set is not None:
        sub_set = args.sub_set
    else:
        sub_set = subset_dict[args.subset_idx]

    if args.model_name is not None:
        model_name = args.model_name
    else:
        model_name = model_dict[args.model_idx]

    # =====================================================
    # path
    # =====================================================

    data_path = data_path_dict[data_name]

    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = (
            f"/2024133105/Workspaces/llms/"
            f"{model_name}"
        )

    output_path = "./results/"

    # =====================================================
    # print
    # =====================================================

    print("=" * 60)
    print("init config")
    print("=" * 60)

    print(f"data_name  : {data_name}")
    print(f"sub_set    : {sub_set}")
    print(f"model_name : {model_name}")
    print(f"model_path : {model_path}")

    print("=" * 60)

    return (
        data_name,
        data_path,
        sub_set,
        model_name,
        model_path,
        output_path
    )


#初始化api参数
def init_api():
    base_url = "https://api.deepseek.com"
    api_key="sk-d05777c30f4a4e7abdb53d2c204f8b6b"
    api_model_name = "deepseek-v4-pro"
    return base_url,api_key,api_model_name

#加载模型和分词器
def load_tokenizer_and_model(model_path,output_attention=False,output_hidden_states=False):
    device="cuda:0"
    # if 'Qwen3' in model_path:
    #     Model_Class=Qwen3ForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        #llm_int8_threshold=6.0,
        #llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        output_attentions=output_attention,
        output_hidden_states=output_hidden_states
        #attn_implementation='eager'
    )
    return tokenizer,model

#找到对应token——id
def find_token_index(offsets,start_char,end_char):
    token_start = token_end = None
    for idx, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            token_start = idx
        if start < end_char <= end:
            token_end = idx
            break
    return token_start,token_end

#调用客户端
def call_ark(query, api_key, base_url, model):
    if OpenAI is None:
        raise ImportError(
            "The openai package is required only for API-backed experiments. "
            "Install it in the active environment before calling call_ark()."
        )
    print("BASE URL:", base_url)
    print("MODEL:", model)
    client = OpenAI(
    # 从环境变量中读取您的方舟API Key
        api_key=api_key,
        base_url=base_url,
        timeout=1800,
    )
    response = client.chat.completions.create(
        # 替换 <Model> 为模型的Model ID
        model=model,
        messages=[
            {"role": "user", "content": query}
        ],
        max_tokens = 16384
    )
    # 当触发深度推理时，打印思维链内容
    cot = None
    if hasattr(response.choices[0].message, 'reasoning_content'):
        cot = response.choices[0].message.reasoning_content
    output = response.choices[0].message.content
    return cot, output

#只替换第n次出现的某个词
def replace_nth(text, old, new, n):
    parts = text.split(old, n)
    if len(parts) <= n:
        return text
    return old.join(parts[:n]) + new + old.join(parts[n:])

#取topk的索引
def get_topk_indices_numpy(lst,top_k=10):
    arr = np.array(lst)
    # argsort默认升序，[::-1]反转得到降序，取前10个
    return np.argsort(arr)[::-1][:top_k].tolist()

def get_low_indices_numpy(lst,top_k=10):
    arr = np.array(lst)
    return np.argsort(arr)[:top_k].tolist()


def get_len(text):
    pieces = re.findall(r'(\S+|\s+)', text)
    return sum(1 for p in pieces if not p.isspace())

def cleanup_model_and_tokenizer(model, tokenizer):
    """清理模型和tokenizer以释放内存"""
    
    # 1. 删除模型引用
    if model is not None:
        del model
    
    # 2. 删除tokenizer引用
    if tokenizer is not None:
        del tokenizer
    
    # 3. 强制进行垃圾回收
    gc.collect()
    
    # 4. 清理PyTorch的CUDA缓存（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
#梯度计算
def analyze_output_impact_gradients(model, tokenizer,original_text, modified_output_target):
    """
    分析如何修改输入来使输出接近目标分布
    
    Args:
        model: 语言模型
        original_text: 原始输入文本
        modified_output_target: 期望的输出目标（可以是分布或特定token）
    """
    inputs = tokenizer(original_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # 获取嵌入
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.requires_grad_(True)
    embeddings.retain_grad()
    
    # 前向传播
    outputs = model(inputs_embeds=embeddings)
    current_logits = outputs.logits
    # 定义输出目标（这里以最后一个输出位置为例）
    target_pos = -1  # 最后一个输出位置
    
    if isinstance(modified_output_target, int):
        # 如果是token ID，创建one-hot目标
        target = torch.zeros_like(current_logits[0, target_pos])
        target[modified_output_target] = 1.0
    else:
        # 如果是分布，直接使用
        target = modified_output_target
    
    # 计算输出差异损失
    loss = F.kl_div(
        F.log_softmax(current_logits[0, target_pos], dim=-1),
        F.log_softmax(target, dim=-1),
        reduction='sum',
        log_target=True
    )
    
    # 反向传播
    loss.backward()
    
    # 分析输入梯度
    input_gradients = embeddings.grad
    
    return {
        'input_gradients': input_gradients,
        'current_output': current_logits,
        'loss': loss.item(),
    }


def hook_attention(model, tokenizer, text):
    """Hook 注意力矩阵并可视化"""
    attention_maps = []
    
    def attention_hook(module, input, output):
        # 保持张量在原始设备上，只分离计算图
        attention_weights = output[1].detach().cpu() 
        attention_maps.append(attention_weights)
    
    # 注册hook
    hooks = []
    # for name, module in model.named_modules():
    #     if "attn" in name and "proj" not in name and "o_proj" not in name:
    #         hook = module.register_forward_hook(attention_hook)
    #         hooks.append(hook)
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn'):
            hook = layer.self_attn.register_forward_hook(attention_hook)
            hooks.append(hook)
    # 前向传播
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # 确保输入在正确的设备上
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 移除hook
    for hook in hooks:
        hook.remove()
    
    # 最后再移动到CPU
    attention_maps = [tensor.cpu() for tensor in attention_maps]
    
    return attention_maps
#获取last_token的attention
def get_attention(model,tokenizer,generation_args,text):
    def get_q_k_hook(all_layer_lists=[]):
        def hook(module, input, output):
            all_layer_lists.append(output.detach().transpose(1, 2))
        return hook
    def get_position_embed_hook(all_position_embed=[]):
        def hook(module, input, output):
            all_position_embed.append(output)
        return hook
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos[:,:,-1,:]) + (rotate_half(q) * sin[:,:,-1,:])
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) :
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    model.eval()
    all_layer_queries=[]
    all_layer_keys=[]
    all_position_embed=[]
    #attention_mask=model.model.attention_mask
    hooks=[]
    hook=model.model.rotary_emb.register_forward_hook(get_position_embed_hook(all_position_embed))
    hooks.append(hook)
    for i,layer in enumerate(model.model.layers):
        q_norm=layer.self_attn.q_norm
        k_norm=layer.self_attn.k_norm
        hook = q_norm.register_forward_hook(get_q_k_hook(all_layer_queries))
        hooks.append(hook)
        hook = k_norm.register_forward_hook(get_q_k_hook(all_layer_keys))
        hooks.append(hook)
    input_ids=tokenizer(text,return_tensors='pt').to(model.device)
    with torch.no_grad():
        output=model(**input_ids,**generation_args)
    for hook in hooks:
        hook.remove()
    cos,sin=all_position_embed[0]
    all_layer_last_token_query_pos_emb=[]
    all_layer_keys_pos_emb=[]
    for i in range(len(all_layer_queries)):
        last_token_query=all_layer_queries[i][:,:,-1,:]
        key_states=all_layer_keys[i]
        #key_states,_=past_key_values[i]
        last_token_query, key_states = apply_rotary_pos_emb(last_token_query, key_states, cos, sin)
        all_layer_last_token_query_pos_emb.append(last_token_query)
        all_layer_keys_pos_emb.append(repeat_kv(key_states,model.model.layers[i].self_attn.num_key_value_groups))
    attn_weights_list=[]
    for i in range(len(all_layer_keys_pos_emb)):
        attn_weights = torch.matmul(all_layer_last_token_query_pos_emb[i].unsqueeze(dim=2), all_layer_keys_pos_emb[i].transpose(2, 3)) * (model.model.layers[i].self_attn.scaling)
        # if attention_mask is not None:
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #     attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(all_layer_last_token_query_pos_emb[i].dtype)
        attn_weights_list.append(attn_weights)
    return attn_weights_list

def get_prediction(text):
    # 先直接匹配单个字母
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1)
    # 再尝试分割后匹配
    split_list = re.split(r'[,;|{:\s]\s*', text)
    for s in split_list:
        s = s.strip()
        if s == 'A': return 'A'
        elif s == 'B': return 'B'
        elif s == 'C': return 'C'
        elif s == 'D': return 'D'
    return None


class InsertTokenProcessor(LogitsProcessor):
    def __init__(self, insert_token_id, insert_position):
        self.insert_token_id = insert_token_id
        self.insert_position = insert_position
    
    def __call__(self, input_ids, scores):
        current_length = input_ids.shape[1]
        
        # 当达到插入位置时，强制下一个token是插入词
        if current_length == self.insert_position:
            # 创建一个全负无穷的分数数组
            forced_scores = scores.clone().fill_(-float("inf"))
            # 只允许插入词的分数为正
            forced_scores[:, self.insert_token_id] = 0
            return forced_scores
        
        return scores

#只保留部分词
def maintain_only_topk(text,replace_words,mask=False):#改inseq_data,改group
    #对text作拆解
    match_before = re.search(r'^(.*?)<think>', text, re.DOTALL)
    if match_before:
        before_think = match_before.group(0)
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL) 
    if match:
        CoT=match.group(1)
    after_think=' </think>'
    sampled_list=replace_words
    if isinstance(sampled_list[0], tuple):
        sampled_phrases=[word[0] for word in sampled_list]
    else:
        sampled_phrases=sampled_list
    #sampled_list = sorted(sampled_list, key=lambda x: x[1], reverse=True)
    if not mask:
        think=' '.join(sampled_phrases)
        changed_text=before_think+think+after_think
    else:
        pieces = re.findall(r'(\S+|\s+)', CoT)#把字符串 text 拆成一个列表 pieces，
        #其中每个元素要么是一段连续的非空白字符（单词、数字、标点等），要么是一段连续的空白字符（空格、制表符 \t、换行 \n 等）
        words=[]
        for p in pieces:
            if not p.isspace():
                words.append(p)                  # 只保留非空白
        word_mask=[]
        for word_tuple in sampled_list:
            count=0
            mask_num=0
            for i in range(0,len(words)):
                if words[i]==word_tuple[0]:
                    count+=1
                    if count==word_tuple[1]:
                        break
            mask_num=i
            word_mask.append((word_tuple[0],mask_num))
        last_mask_num=len(words)-mask_num-1#计算最后要补足多少个mask,其中mask_num取循环最后一次的mask_num
        think=''
        for idx in range(len(word_mask)):
            if idx==0:
                think+='... '*word_mask[idx][1]
            else:
                think+='... '*(word_mask[idx][1]-word_mask[idx-1][1]-1)
            think+=word_mask[idx][0]
            think+=' '
        think+='... '*(last_mask_num-1)
        think+='...'
        changed_text=before_think+think+after_think#'\n'模拟真实输出情况
    return changed_text

#用于将words合并为句子同时合并word-level的attention
def merge_sentences_with_indices_v2(words, end_punctuations=None):
    """
    增强版：支持多种结束标点
    
    Args:
        words: 列表，元素为 (word, index) 元组
        end_punctuations: 结束标点列表，默认为 ['.', '。', '!', '！', '?', '？']
    
    Returns:
        合并后的 (sentence, total_index) 列表
    """
    if end_punctuations is None:
        end_punctuations = ['.', '。', '!', '！', '?', '？']
    
    merged = []
    current_sentence = []
    current_indices = []
    
    for word, index in words:
        current_sentence.append(word)
        current_indices.append(index)
        
        # 检查是否以任何结束标点结尾
        if any(word.endswith(punc) for punc in end_punctuations):
            # 合并句子
            sentence = ' '.join(current_sentence)
            total_index = sum(current_indices)
            merged.append((sentence, total_index))
            
            # 重置
            current_sentence = []
            current_indices = []
    
    # 处理剩余部分
    if current_sentence:
        sentence = ' '.join(current_sentence)
        total_index = sum(current_indices)
        merged.append((sentence, total_index))
    
    return merged

#template
MCQ_TEMPLATE = '''
    {q}
    Please tell me the option of the correct answer, display only the option letters and put it in \\boxed{{}} in the answer field, for example, "answer": "\\boxed{{C}}": 
    A. {o1}
    B. {o2}
    C. {o3}
    D. {o4}
    '''
MCQ_TEMPLATE_FOR_LOGITS = '''
    {q}
    Please tell me the option of the correct answer: 
    A. {o1}
    B. {o2}
    C. {o3}
    D. {o4}
    Please provide the option of the answer directly after giving the thinking process
    '''
Synonyms_TEMPLATE='''
    Please strictly follow the following instructions.
    Generate a synonym for each of the following phrases, don't miss a single word, and only output the original phrase and the synonym in dictionary format, separated by colons for each pair of original phrases and synonyms
    ### Example:
    ['happy', 'fast']  
    {{'happy': 'joyful', 'fast': 'quick'}}

    ### Now do it for:
    {q}
    '''
Antonyms_TEMPLATE='''
    Please strictly follow the following instructions.
    Generate an antonym for each of the following phrases, don't miss a single word, and only output the original phrase and the antonym in dictionary format, separated by colons for each pair of original phrases and antonyms
    ### Example:
    ['happy', 'fast']  
    {{'happy': 'sad', 'fast': 'slow'}}

    ### Now do it for:
    {q}
    '''
    #ASSIST_TEMPLATE = 'Please think through and analyze the pros and cons of each option step by step. At the end, clearly state the best choice, using the following format:Answer: A (or B/C/D) '
    #THINK_TEMPLATE='<think>{t}</think> \n The correct answer is'
THINK_TEMPLATE='<think>{t}</think>'
THINK_TEMPLATE_CONTINUE='<think>{t}'
