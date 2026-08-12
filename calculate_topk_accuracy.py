import json
from pathlib import Path


# 默认读取本脚本所在项目中的 output 文件夹。
# 如果结果实际位于其他目录，只需修改这里，例如："origin_topk_result/v3"。
OUTPUT_DIR = "output"
MODEL_NAME = "Qwen3-8B"
DATASET_NAME = "mmlu_redux"
TOP_PERCENTAGE = 0.1
FILE_PATTERN = (
    f"{MODEL_NAME}-{DATASET_NAME}-*-att-right-"
    f"top_percentage{TOP_PERCENTAGE}.json"
)


def sample_is_correct(sample):
    """优先读取 correct；没有该字段时比较 prediction 和 truth。"""
    if "correct" in sample:
        correct = sample["correct"]
        if isinstance(correct, bool):
            return correct
        if isinstance(correct, (int, float)) and correct in (0, 1):
            return bool(correct)
        if isinstance(correct, str) and correct.strip().lower() in {
            "0", "1", "false", "true"
        }:
            return correct.strip().lower() in {"1", "true"}
        raise ValueError(f"无法识别的 correct 值：{correct!r}")

    if "prediction" not in sample or "truth" not in sample:
        raise KeyError("样本缺少 correct，且没有完整的 prediction/truth 字段")

    prediction = str(sample["prediction"]).strip().upper()
    truth = str(sample["truth"]).strip().upper()
    return prediction == truth


def calculate_file_accuracy(json_path):
    with json_path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list):
        raise ValueError("JSON 顶层必须是样本列表")

    total = len(samples)
    correct = sum(sample_is_correct(sample) for sample in samples)
    accuracy = correct / total if total else 0.0
    return correct, total, accuracy


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / OUTPUT_DIR

    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"结果目录不存在：{output_dir}\n"
            "请修改脚本顶部的 OUTPUT_DIR。"
        )

    json_files = sorted(output_dir.glob(FILE_PATTERN))
    if not json_files:
        raise FileNotFoundError(
            f"在 {output_dir} 中没有找到匹配 {FILE_PATTERN!r} 的文件"
        )

    total_correct = 0
    total_samples = 0

    print(f"找到 {len(json_files)} 个结果文件：")
    for json_path in json_files:
        try:
            correct, total, accuracy = calculate_file_accuracy(json_path)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
            print(f"[跳过] {json_path.name}: {error}")
            continue

        total_correct += correct
        total_samples += total
        print(
            f"{json_path.name}: "
            f"{correct}/{total}, accuracy={accuracy:.4f} ({accuracy:.2%})"
        )

    overall_accuracy = (
        total_correct / total_samples if total_samples else 0.0
    )
    print("\n总体结果：")
    print(
        f"{total_correct}/{total_samples}, "
        f"accuracy={overall_accuracy:.4f} ({overall_accuracy:.2%})"
    )


if __name__ == "__main__":
    main()
