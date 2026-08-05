
import os
import subprocess
import logging
import time


# =========================================================
# 配置
# =========================================================

DATA_NAME = "mmlu_redux"

MODEL_NAME = "Qwen3-8B"

ROOT = "."


# =========================================================
# 自动发现 subset
# =========================================================

def discover_subsets():

    result_dir = f"{ROOT}/results/{DATA_NAME}"

    subset_list = []

    for file in os.listdir(result_dir):

        if (
            file.startswith(MODEL_NAME)
            and file.endswith("_filter_right.json")
        ):

            subset = (
                file.replace(
                    f"{MODEL_NAME}_",
                    ""
                )
                .replace(
                    "_filter_right.json",
                    ""
                )
            )

            subset_list.append(subset)

    subset_list = sorted(subset_list)

    return subset_list


# =========================================================
# 判断attention是否已经完成
# =========================================================

def attention_finished(subset):

    att_path = (
        f"{ROOT}/results/{DATA_NAME}/att_grad_before/"
        f"{MODEL_NAME}_{subset}_rate1_all_att_list.json"
    )

    words_path = (
        f"{ROOT}/results/{DATA_NAME}/att_grad_before/"
        f"{MODEL_NAME}_{subset}_rate1_words_list.json"
    )

    return (
        os.path.exists(att_path)
        and os.path.exists(words_path)
    )


# =========================================================
# 判断实验是否已经完成
# =========================================================

def experiment_finished(subset):

    result_path = (
        f"{ROOT}/results/two_experiments/"
        f"{DATA_NAME}/{MODEL_NAME}/{subset}/"
        f"experiment_B_results.json"
    )

    return os.path.exists(result_path)


# =========================================================
# 跑attention extraction
# =========================================================

def run_attention(subset):

    print("=" * 60)
    print(f"[Attention] Running subset: {subset}")
    print("=" * 60)

    cmd = [
        "python3",
        "get_att.py",
        "--data_name", DATA_NAME,
        "--sub_set", subset,
        "--model_name", MODEL_NAME
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"Attention extraction failed: {subset}"
        )


# =========================================================
# 跑two experiments
# =========================================================

def run_experiments(subset):

    print("=" * 60)
    print(f"[Experiments] Running subset: {subset}")
    print("=" * 60)

    cmd = [
        "python3",
        "run_two_experiments.py",
        "--data_name", DATA_NAME,
        "--sub_set", subset,
        "--model_name", MODEL_NAME
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"Experiments failed: {subset}"
        )


# =========================================================
# 主程序
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # 日志
    # -----------------------------------------------------

    os.makedirs("./log", exist_ok=True)

    log_file = (
        f"./log/pipeline_"
        f"{MODEL_NAME}_{DATA_NAME}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                log_file,
                encoding="utf-8"
            ),
            logging.StreamHandler()
        ]
    )

    # -----------------------------------------------------
    # subset discovery
    # -----------------------------------------------------

    subset_list = discover_subsets()

    logging.info(
        f"发现 {len(subset_list)} 个 subsets"
    )

    for subset in subset_list:

        logging.info(
            f"开始处理 subset: {subset}"
        )

        start_time = time.time()

        try:

            # =================================================
            # step1: attention
            # =================================================

            if attention_finished(subset):

                logging.info(
                    f"[Skip] attention 已完成: {subset}"
                )

            else:

                run_attention(subset)

                logging.info(
                    f"[Done] attention 完成: {subset}"
                )

            # =================================================
            # step2: experiments
            # =================================================

            if experiment_finished(subset):

                logging.info(
                    f"[Skip] experiments 已完成: {subset}"
                )

            else:

                run_experiments(subset)

                logging.info(
                    f"[Done] experiments 完成: {subset}"
                )

            elapsed = (
                time.time() - start_time
            ) / 60

            logging.info(
                f"subset {subset} 完成 "
                f"耗时 {elapsed:.2f} min"
            )

        except Exception as e:

            logging.error(
                f"subset {subset} 失败: {e}"
            )

            continue

    logging.info("=" * 60)
    logging.info("全部 pipeline 完成！")
    logging.info("=" * 60)
