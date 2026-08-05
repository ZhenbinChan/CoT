
from utils import *

import json
import os
import re
import logging
import sys

import torch
from tqdm import tqdm


# =========================================================
# 获取 attention
# =========================================================
def get_att(
    filter_data,
    model,
    tokenizer,
    generation_args
):

    words_list = []
    all_att_list = []

    model.eval()

    for idx in tqdm(range(len(filter_data))):

        text = filter_data[idx]['all_output_text']

        # tokenizer offset
        encoding = tokenizer(
            text,
            return_offsets_mapping=True
        )

        offsets = encoding['offset_mapping']

        input_length = len(
            filter_data[idx]['input_text']
        )

        # =========================================================
        # 找 boxed 答案
        # =========================================================

        match = re.search(
            r"\\boxed\{([^}]+)\}",
            text[input_length:],
            re.DOTALL
        )

        if not match:
            continue

        start = match.start()

        # 截断到 boxed 前
        # 保证 last token 是答案前一个 token
        sub_text = text[:start + 7 + input_length]

        # =========================================================
        # 获取 attention
        # =========================================================

        with torch.no_grad():

            output_attention = get_attention(
                model,
                tokenizer,
                generation_args,
                sub_text
            )

            # [layer, batch, head, seq, seq]
            att = torch.stack(
                output_attention,
                dim=0
            )

            # 对 layer 平均
            att = att.mean(dim=0)

            # [head, seq, seq]
            att_cpu = att.cpu()

            # 对 head 平均
            # [seq, seq]
            last_token_att = att_cpu.mean(
                dim=1
            ).squeeze()

            del output_attention
            del att

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # =========================================================
        # word span
        # =========================================================

        pieces = re.findall(
            r'(\S+|\s+)',
            text
        )

        words = []
        word_spans = []

        start = 0

        for p in pieces:

            end = start + len(p)

            if not p.isspace():

                words.append(p)

                word_spans.append((start, end))

            start = end

        words_list.append(words)

        # =========================================================
        # word-level attention
        # =========================================================

        att_list = []

        for start, end in word_spans:

            token_start, token_end = find_token_index(
                offsets=offsets,
                start_char=start,
                end_char=end
            )

            att_value = last_token_att[
                token_start:token_end + 1
            ].sum()

            att_list.append(
                att_value.item()
            )

        all_att_list.append(att_list)

        del last_token_att

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_att_list, words_list


# =========================================================
# main
# =========================================================
if __name__ == '__main__':

    root = "."

    (
        data_name,
        data_path,
        sub_set,
        model_name,
        model_path,
        output_path
    ) = init()

    # =========================================================
    # 日志
    # =========================================================

    log_file = (
        f"./log/"
        f"{model_name}-attention-"
        f"{data_name}_{sub_set}.log"
    )

    log_dir = os.path.dirname(log_file)

    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_file,
                encoding='utf-8'
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("开始加载模型...")

    # =========================================================
    # load model
    # =========================================================

    tokenizer, model = load_tokenizer_and_model(
        model_path=model_path
    )

    model.eval()

    logging.info(
        f"finish load tokenizer and model "
        f"({model_name})!"
    )

    # =========================================================
    # generation args
    # =========================================================

    generation_args = {
        "do_sample": False,
        "max_new_tokens": 1024,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        "repetition_penalty": 1.2
    }

    # =========================================================
    # load data
    # =========================================================

    logging.info("开始加载数据...")

    with open(
        f"{root}/results/{data_name}/"
        f"{model_name}_{sub_set}_filter_right.json"
    ) as f:

        filter_data = json.load(f)

    logging.info(
        f"finish load data "
        f"({data_name}-{sub_set})!"
    )

    # =========================================================
    # get attention
    # =========================================================

    logging.info("开始获取 attention...")

    all_att_list, words_list = get_att(
        filter_data=filter_data,
        model=model,
        tokenizer=tokenizer,
        generation_args=generation_args
    )

    logging.info(
        "finish get attention!"
    )

    # =========================================================
    # 保存
    # =========================================================

    save_dir = (
        f"{root}/results/"
        f"{data_name}/att_grad_before"
    )

    os.makedirs(save_dir, exist_ok=True)

    # 注意：
    # 这里保留 rate1 命名
    # 为了兼容旧版 run_two_experiments.py

    rate = 1

    # =========================================================
    # 保存 attention
    # =========================================================

    att_path = (
        f"{save_dir}/"
        f"{model_name}_{sub_set}_"
        f"rate{rate}_all_att_list.json"
    )

    with open(
        att_path,
        'w',
        encoding='utf-8'
    ) as f:

        json.dump(
            all_att_list,
            f,
            ensure_ascii=False,
            indent=4
        )

    logging.info(
        f"save attention to {att_path}"
    )

    # =========================================================
    # 保存 words
    # =========================================================

    words_path = (
        f"{save_dir}/"
        f"{model_name}_{sub_set}_"
        f"rate{rate}_words_list.json"
    )

    with open(
        words_path,
        'w',
        encoding='utf-8'
    ) as f:

        json.dump(
            words_list,
            f,
            ensure_ascii=False,
            indent=4
        )

    logging.info(
        f"save words to {words_path}"
    )

    logging.info(
        f"finish attention extraction for "
        f"{data_name}_{sub_set}_{model_name}"
    )
