# CoT Faithfulness 实验代码说明

## 1. 项目研究问题

本项目研究大语言模型显式思维链（Chain-of-Thought，CoT）的忠实性，核心问题是：

> 模型输出的 CoT 是否真正参与了最终答案的形成，以及 attention 能否可靠地识别 CoT 中对答案具有因果作用的内容。

项目不只比较“使用 CoT”和“不使用 CoT”的准确率，还会对 CoT 中的词或句子进行替换、删除和保留，再观察模型答案是否发生变化。主要研究问题包括：

1. 高 attention 的 CoT 内容是否比低 attention 或随机内容更影响答案？
2. 同义词替换和反义词替换是否会带来不同的答案变化？
3. 替换 CoT 后，模型的 attention 是否发生有意义的迁移？
4. 只保留少量高 attention 词或句子时，模型能否维持原来的推理能力？
5. 模型自己声称的重要句子是否与其内部 attention 一致？

项目的基本实验范式是：

```text
构造 CoT 可能真正有用的样本
        ↓
测量最终答案位置对 CoT 的 attention
        ↓
替换、删除或只保留部分 CoT
        ↓
重新计算答案和 attention
        ↓
比较 attention、答案正确率和模型自我解释
```

## 2. 项目结构

仓库中存在两套并行的实验链路：

```text
共同数据阶段
1_prepare_dataset.py
        │
        └── results/.../*_filter_right.json
               │
       ┌───────┴────────────────────────┐
       │                                │
当前句子级链路                    旧版词级干预链路
get_att.py                       2_get_attention_and_gradient.py
       │                                │
词级 attention                  attention + 同/反义词替换
       │                                │
run_two_experiments.py           3_get_attention_after_replacing.py
       │                                │
高/低/随机句子和模型自选句       替换后 attention 与答案变化
```

`get_att.py` 和 `2_get_attention_and_gradient.py` 是两个不同实验链路的入口，不是必须连续运行的两个阶段。它们都会向 `results/{data_name}/att_grad_before/` 写入原始 attention 文件。

## 3. 数据准备：`1_prepare_dataset.py`

### 3.1 功能

`1_prepare_dataset.py` 首先加载选择题数据，并分别运行：

- thinking 模式：允许模型生成 `<think>...</think>`；
- no-thinking 模式：关闭模型的 thinking。

随后只保留满足下面条件的数据：

```text
thinking 模式答对 && no-thinking 模式答错
```

这样可以将后续实验集中在显式推理至少与正确答案存在行为关联的样本上。

### 3.2 主要函数

- `load_data()`：加载 Hugging Face 数据集或本地 CSV。
- `handle_data()`：处理 MMLU-Redux，只保留有效题目并修正错误标注。
- `model_output()`：分别生成 thinking 和 no-thinking 输出。
- `data_filter()`：筛选 thinking 正确、no-thinking 错误的样本。

### 3.3 输出文件

```text
datasets/{data_name}/{sub_set}_new.json

results/{data_name}/
├── {model_name}_{sub_set}_think.json
├── {model_name}_{sub_set}_nothink.json
└── {model_name}_{sub_set}_filter_right.json
```

`filter_right.json` 是后续所有重点实验的共同输入，主要字段包括：

```text
question
choices
CoT
input_text
output_text
all_output_text
prediction
nothink_prediction
truth
```

## 4. Attention 的计算方式

三个重点 attention 脚本最终都调用 `utils.py` 中的 `get_attention()`。

该函数没有直接读取 Transformers 的 `output_attentions`，而是：

1. 为每一层的 `q_norm` 和 `k_norm` 注册 forward hook；
2. 获取各层的 query 和 key；
3. 手工应用 RoPE；
4. 取最后一个 token 的 query；
5. 重建最后一个 token 对所有前文 token 的 attention；
6. 对所有层和 attention head 取平均；
7. 根据 tokenizer 的字符偏移，将 subword attention 求和为空白分隔的词级 attention。

对第 `l` 层、第 `h` 个 head，可以概括为：

```text
A_lh = softmax(q_last · K^T / sqrt(d))
```

最终词级值是相应 token attention 的求和，并在 layer/head 维度上取平均。

输入文本会被截断到最终答案的 `\boxed{`。因此最后一个位置代表“答案字母尚未出现时”的决策位置，attention 表示该位置对题目和 CoT 前文的关注程度。

代码还将第一个 token 的较大 attention 视为 attention sink。在部分比例计算中，分母使用 `1 - first_token_attention`。

## 5. `get_att.py`：当前原始 Attention 提取脚本

### 5.1 定位

这是当前句子级实验链路使用的精简版 attention 提取器。它不调用外部替换 API，也不进行 CoT 干预，只计算原始输出中每个词的 attention。

### 5.2 调用逻辑

```text
main
 ├── utils.init()
 ├── utils.load_tokenizer_and_model()
 ├── 读取 filter_right.json
 ├── get_att()
 │    ├── 定位最终答案的 \boxed{
 │    ├── 截断到答案字母出现之前
 │    ├── utils.get_attention()
 │    ├── tokenizer offset → 字符/token 对齐
 │    └── token attention → whitespace-word attention
 └── 保存 all_att_list 和 words_list
```

### 5.3 输入

```text
results/{data_name}/{model_name}_{sub_set}_filter_right.json
```

### 5.4 输出

```text
results/{data_name}/att_grad_before/
├── {model_name}_{sub_set}_rate1_all_att_list.json
└── {model_name}_{sub_set}_rate1_words_list.json
```

两份列表按位置对齐：

```text
words_list[i][j]     第 i 条样本的第 j 个空白分隔词
all_att_list[i][j]   对应词从答案决策位置获得的 attention
```

文件名中的 `rate1` 是为了兼容旧版 `run_two_experiments.py`，当前 `get_att.py` 不进行任何替换。

代码和部分注释会将保存结果称为 token-level attention，但保存前已经按照字符跨度聚合成了 whitespace-word level attention。

## 6. `2_get_attention_and_gradient.py`：旧版 Attention 与词替换实验

### 6.1 定位

这是旧版词级干预链路的核心脚本，负责：

- 计算替换前 attention；
- 随机选择 CoT 内容；
- 替换高 attention 词；
- 生成同义词和反义词；
- 让原模型从修改后的 CoT 继续生成答案；
- 记录答案是否发生变化。

虽然文件名中包含 `gradient`，但当前版本的梯度计算和梯度文件保存代码均已注释，实际只执行 attention 实验。`att_grad_before` 目录名也是历史命名。

### 6.2 主调用流程

```text
读取 filter_right.json
        │
        ▼
get_att()
 ├── 计算原始词级 attention
 ├── 选择随机替换词
 ├── 计算整个 CoT 的 attention
 └── 统计被选中词在替换前的 attention
        │
        ├── random_replace(synonym)
        ├── random_replace(antonym)
        │
        ├── topk_replace(synonym, 10%/20%/30%)
        └── topk_replace(antonym, 10%/20%/30%)
```

### 6.3 `extract_and_replace_phrases()`

该函数负责选择替换内容和调用外部 API。

随机模式下：

1. 提取 `<think>...</think>`；
2. 同时提取 `</think>` 到 `\boxed` 之间的 explanation；
3. 使用 spaCy 进行英文词性标注；
4. 选择形容词和副词；
5. 使用固定随机种子抽样；
6. 调用外部 API 生成同义词或反义词；
7. 生成实际替换文本和带方括号标记的替换文本。

```text
replace_text         实际送回模型的文本
replace_text_w_mark  用 [replacement] 标记替换位置的文本
```

需要注意，代码中的“随机替换”并不是从所有 CoT 词中均匀抽样，而是从 spaCy 判断出的形容词和副词中抽样。默认 `rate=1` 时会选择全部去重后的候选词。

Top-k 模式下，函数接收 `(word, occurrence_number)`，只替换该词的第 N 次出现。

### 6.4 `get_att()`

该函数同时计算：

- 所有词的原始 attention；
- 整个 CoT 的 attention；
- 随机选中替换词的 attention；
- 被替换词 attention 占 CoT attention 的比例。

额外摘要文件中的字段为：

```json
{
  "change_attention": "被选中词的 attention 总和",
  "CoT_att_percentage": "CoT attention 占有效总 attention 的比例",
  "change_attention_percentage_in_CoT": "被选中词 attention 占 CoT attention 的比例"
}
```

### 6.5 `random_replace()`

该函数重新调用替换函数，将修改后的 CoT 作为前缀交给原模型继续生成，并记录：

- 新答案；
- 新答案是否正确；
- 原始答案；
- 被替换词；
- 同义词或反义词；
- 替换比例；
- 被替换内容在原始文本中的 attention。

### 6.6 `topk_replace()`

该函数：

1. 在 CoT 和 explanation 范围内选择 attention 最高的词；
2. 分别取前 10%、20%、30%；
3. 对这些词进行同义词或反义词替换；
4. 让模型从修改后的推理继续生成答案；
5. 保存答案正确性和被替换词的 attention 占比。

### 6.7 输出

```text
results/{data_name}/{model_name}_{sub_set}_rate1_right_att.json

results/{data_name}/att_grad_before/
├── {model_name}_{sub_set}_rate1_all_att_list.json
└── {model_name}_{sub_set}_rate1_words_list.json

final_results/replace_results/
├── {model_name}-{data_name}-{sub_set}-synonym-rate1_random_right.json
├── {model_name}-{data_name}-{sub_set}-antonym-rate1_random_right.json
├── {model_name}-{data_name}-{sub_set}-synonym-att-right-top_percentage0.1.json
├── {model_name}-{data_name}-{sub_set}-antonym-att-right-top_percentage0.1.json
└── 对应 0.2、0.3 文件
```

## 7. `3_get_attention_after_replacing.py`：替换后的 Attention

### 7.1 定位

该脚本依赖 `2_get_attention_and_gradient.py` 生成的全部 synonym/antonym replacement JSON，用于重新测量干预后文本的 attention。

它不会自动运行第二阶段，因此必须保证 replacement 文件已经存在。

### 7.2 调用逻辑

```text
读取 replacement JSON
        │
        ▼
重新构造替换后的完整推理文本
        │
        ▼
get_att()
 ├── 重新计算答案位置的 attention
 ├── 找到带 [ ] 的替换词
 └── 计算整个 CoT 的新 attention
        │
        ▼
get_random_results() / get_top_results()
        │
        ├── 添加 att_after_replacing
        └── 添加 att_percentage_in_CoT_after_replacing
```

脚本构造的文本大致为：

```text
原始题目和 chat 前缀
+ 带方括号标记的替换后 CoT
+ 带方括号标记的 explanation
+ 重新生成的 boxed 答案
```

### 7.3 输出

中间结果保存到：

```text
results/{data_name}/att_grad_after/
├── ..._all_att_list_{synonyms|antonyms}.json
├── ..._all_CoT_att_list_{synonyms|antonyms}.json
└── ..._words_list_{synonyms|antonyms}.json
```

同时，脚本会原地更新 `final_results/replace_results/*.json`，增加：

```json
{
  "att_after_replacing": "替换词在干预后的 attention 总和",
  "att_percentage_in_CoT_after_replacing": "替换词新 attention 占新 CoT attention 的比例"
}
```

需要注意，该脚本是在带 `[replacement]` 方括号标记的文本上重新计算 attention。因此替换后 attention 同时受到语义变化和额外方括号 token 的影响，这是解释结果时需要考虑的潜在混淆因素。

## 8. 当前句子级实验：`run_two_experiments.py`

当前推荐链路在 `get_att.py` 之后运行 `run_two_experiments.py`。

该脚本首先将词级 attention 聚合到句子级，然后进行两个实验。

### 8.1 实验 A：attention 选句

分别构造：

- 完整 CoT；
- 无 CoT；
- 只保留高 attention 的 10%、20%、30% 句子；
- 只保留低 attention 的相同比例句子；
- 随机保留相同比例句子。

脚本不重新生成完整长文本，而是读取候选答案 A/B/C/D 在最后位置的 logits，以概率最大的选项作为预测。

### 8.2 实验 B：模型自选句

1. 将原始 CoT 提供给模型；
2. 要求模型选择最多五个最重要句子；
3. 使用 Rouge-L 将模型返回文本映射回原始 CoT 句子；
4. 只保留这些句子并重新判断答案；
5. 统计模型自选句的正确率和 attention 占比。

### 8.3 输出

```text
results/two_experiments/{data_name}/{model_name}/{sub_set}/
├── merge_result.json
├── experiment_A_results.json
├── model_selected_sentences.json
└── experiment_B_results.json
```

## 9. 推荐运行顺序

### 9.1 当前句子级实验：单个 subset

根据集群使用原则，先激活环境并申请计算节点：

```bash
conda activate cot
srun -G 2 --cpus-per-task=2 -t 120:00:00 --pty bash -i
```

进入项目目录：

```bash
cd /2024133105/Workspaces/CoT-faithfulness
```

依次运行：

```bash
python 1_prepare_dataset.py \
  --data_name mmlu_redux \
  --sub_set global_facts \
  --model_name Qwen3-0.6B

python get_att.py \
  --data_name mmlu_redux \
  --sub_set global_facts \
  --model_name Qwen3-0.6B

python run_two_experiments.py \
  --data_name mmlu_redux \
  --sub_set global_facts \
  --model_name Qwen3-0.6B
```

运行顺序为：

```text
1_prepare_dataset.py
        ↓
get_att.py
        ↓
run_two_experiments.py
```

### 9.2 当前句子级实验：批量运行

先批量生成 `filter_right.json`：

```bash
DATA_NAME=mmlu_redux \
MODEL_NAME=Qwen3-8B \
bash scripts/1_data_preprocessing.sh
```

然后运行：

```bash
python run_pipeline.py
```

`run_pipeline.py` 会：

1. 自动发现已有的 `*_filter_right.json`；
2. 缺少 attention 时运行 `get_att.py`；
3. 缺少最终实验时运行 `run_two_experiments.py`；
4. 自动跳过已完成的 subset；
5. 某个 subset 失败时记录日志并继续运行其他 subset。

当前 `run_pipeline.py` 将配置硬编码为：

```python
DATA_NAME = "mmlu_redux"
MODEL_NAME = "Qwen3-8B"
```

如果需要运行其他数据集或模型，需要修改这两个常量。环境变量不会覆盖它们。

### 9.3 旧版词替换链路

```bash
conda activate cot
srun -G 2 --cpus-per-task=2 -t 120:00:00 --pty bash -i

cd /2024133105/Workspaces/CoT-faithfulness

python 1_prepare_dataset.py \
  --data_name mmlu_redux \
  --sub_set global_facts \
  --model_name Qwen3-0.6B

python 2_get_attention_and_gradient.py \
  --data_name mmlu_redux \
  --sub_set global_facts \
  --model_name Qwen3-0.6B

python 3_get_attention_after_replacing.py \
  --data_name mmlu_redux \
  --sub_set global_facts \
  --model_name Qwen3-0.6B
```

运行顺序为：

```text
1_prepare_dataset.py
        ↓
2_get_attention_and_gradient.py
        ↓
3_get_attention_after_replacing.py
        ↓
10_analyze.py（最终统计，可选）
```

## 10. 其他编号脚本的依赖关系

```text
原始 attention
 ├── 4_maintain_only_topk_words.py
 │     只保留高/低/随机 attention 词
 │
 ├── 6_maintain_sentences.py
 │     将词级 attention 合并为句子级并进行保留实验
 │
 ├── 7_hidden_states_analyze.py
 │     提取 hidden states，执行 TSNE/UMAP 表征分析
 │       └── 8_KNN_maintain.py
 │             根据隐藏空间近邻选择和保留内容
 │
 └── 9_metric.py
       比较 attention 排名与模型自述的重要句子

6_maintain_sentences.py + 9_metric.py
 └── 12_maintain_model_selected_sentences.py
       验证模型自选句子的实际答题效果

上述实验结果
 └── 10_analyze.py
       汇总答案变化和各类对照实验
```

`5_recover_from_mistakes.py` 依赖反义词替换结果，研究模型在受到误导后是否能够恢复正确答案。

## 11. 重要实现注意事项

### 11.1 Gradient 当前未启用

`2_get_attention_and_gradient.py` 和 `3_get_attention_after_replacing.py` 中的梯度代码已经注释。当前结果不能解释为 attention-gradient 联合实验，`att_grad_before` 和 `att_grad_after` 只是历史目录名。

### 11.2 模型路径和 GPU 固定

`utils.init()` 将模型路径构造成：

```text
/home/chenzhb/Workspaces/LLMs/{model_name}
```

需要在运行前确认该路径存在。模型加载代码固定使用 `cuda:0`，申请两张 GPU 不会自动实现双卡并行。

### 11.3 模型不是按 4-bit 方式加载

虽然 `utils.py` 创建了 `BitsAndBytesConfig`，但该配置没有传给 `from_pretrained()`。当前实际使用 FP16 加载，而不是注释和变量可能暗示的 4-bit 加载。

### 11.4 外部依赖

旧版替换实验需要：

- spaCy；
- `en_core_web_sm`；
- 外部大模型 API；
- 能够访问 Hugging Face 数据和本地模型目录。

### 11.5 凭据安全

当前代码中存在硬编码的 Hugging Face token 和 API key。共享或提交代码前应：

1. 撤销现有凭据；
2. 创建新凭据；
3. 从环境变量读取；
4. 不要把 token 写入源代码、日志或 JSON 结果。

### 11.6 数据对齐风险

`get_att.py` 遇到没有合法 `\boxed{}` 的样本时会直接 `continue`。这可能导致 attention/words 列表比 `filter_right.json` 更短，进而使后续按索引读取时发生错位。运行前应确认筛选数据全部包含合法 boxed 答案，或者在代码中为失败样本保存 `None` 占位。

### 11.7 替换函数的文本匹配不是严格 token 匹配

旧版替换使用字符串查找和 `str.replace()`。可能出现：

- 一个词作为另一个词的子串而被替换；
- 同一个词的多次出现被一起替换；
- spaCy 将标签、公式或标点误判为候选词；
- API 返回格式异常时，同义词/反义词与原词错位。

解释实验结果时需要检查实际的 `replaced_phrases` 和生成文本，而不能只根据替换比例判断。

### 11.8 Attention 不等于因果解释

项目计算的是答案位置对前文的平均 attention。高 attention 表示模型在这个位置对相应内容赋予较高权重，但它本身不能证明该内容对答案具有因果作用。

真正用于检验 faithfulness 的是后续行为干预：

- 高 attention 与低 attention 内容的替换差异；
- 同义词与反义词干预差异；
- 删除或只保留内容后的准确率；
- attention 排名与模型自选内容的对齐程度。

因此，最终科研结论应结合 attention 指标和干预后的答案变化共同给出。

## 12. CoT Case Analysis Experiments

这组 case analysis 是独立于现有 v1–v4 的新实验链路。它不会修改旧脚本，也不会覆盖旧结果，默认输出到 `final_results/case_study/`。三个研究问题分别是：

1. 不提供 CoT 时，模型仅凭题目本身能达到多高的准确率？
2. 保留 10%、20%、30% 高 attention 内容带来的增益，有多少来自答案字母、正确选项词或选项位置泄漏？
3. 人工删除或反事实修改 CoT 中的线索后，答案及答案置信度如何变化？

所有新实验使用完全相同的答案读取协议：prompt 以 `\boxed{` 结尾，关闭采样，最多生成 8 tokens，不允许模型重新生成 explanation。准确率以实际生成并解析出的 `prediction` 计算。

### 12.1 公共输入和概率字段

No-CoT 和 Top-k 实验读取：

```text
results/{data_name}/{model_name}_{subset}_filter_right.json
```

Top-k 还读取：

```text
results/{data_name}/att_grad_before/
├── {model_name}_{subset}_rate1_all_att_list.json
└── {model_name}_{subset}_rate1_words_list.json
```

如果这两个 attention 文件不存在，Top-k 启动脚本会先调用 `get_att.py`。新 runner 会严格检查 filter、attention 和 words 的样本数以及每条样本的 word 数；任何错位都会立即报错，而不会继续产生不可解释的结果。

每条预测同时保存 `\boxed{` 后第一个 token 位置的 A/B/C/D 概率：

- `option_probabilities_raw`：四个字母在完整模型词表 softmax 中的原始概率；四项之和通常小于 1。
- `option_probabilities_normalized`：只在 A/B/C/D 四项内部重新归一化后的概率；四项之和为 1。
- `option_probability_mass`：四个原始概率之和。
- `option_argmax_prediction`：四项归一化概率的最大项，仅用于和生成答案做一致性检查。
- `truth_probability_raw` 和 `truth_probability_normalized`：正确标签对应的两种概率。
- `option_top1_probability` 和 `option_top1_top2_margin`：四项内部 top-1 置信度及 top-1/top-2 间隔。

### 12.2 No-CoT direct-answer baseline

`13_no_cot_direct_answer.py` 完全忽略原始 CoT 和 explanation，实际输入为：

```text
sample["input_text"] + " \\boxed{"
```

每个 subset 只产生一份 baseline，不会人为复制成 0.1/0.2/0.3 三份。运行一个指定 subset：

```bash
DATA_NAME=mmlu_redux \
MODEL_NAME=Qwen3-8B \
MODEL_PATH=/path/to/Qwen3-8B \
bash scripts/4_no_cot_direct_answer.sh global_facts
```

不传 subset 时，会自动发现当前 `DATA_NAME` 和 `MODEL_NAME` 的全部 `*_filter_right.json`：

```bash
MODEL_NAME=Qwen3-8B MODEL_PATH=/path/to/Qwen3-8B \
bash scripts/4_no_cot_direct_answer.sh
```

输出结构为：

```text
final_results/case_study/no_cot/{data_name}/{model_name}/
├── {subset}/
│   ├── records.json
│   ├── good_cases.json
│   ├── bad_cases.json
│   └── summary.json
├── aggregate_summary.json
└── aggregate_summary.csv
```

`sample_index` 用于和后续三个 ratio 的相同样本对齐。

### 12.3 五个 Top-k 答案泄漏条件

`14_topk_answer_leakage.py` 是五个条件共享的 Python runner。每个启动脚本只运行一个固定条件，并在同一次模型加载中依次运行 0.1、0.2、0.3：

```bash
MODEL_NAME=Qwen3-8B MODEL_PATH=/path/to/Qwen3-8B \
bash scripts/5-1_topk_letter_kept.sh global_facts

MODEL_NAME=Qwen3-8B MODEL_PATH=/path/to/Qwen3-8B \
bash scripts/5-2_topk_letter_removed.sh global_facts

MODEL_NAME=Qwen3-8B MODEL_PATH=/path/to/Qwen3-8B \
bash scripts/5-3_topk_correct_option_words_removed.sh global_facts

MODEL_NAME=Qwen3-8B MODEL_PATH=/path/to/Qwen3-8B \
bash scripts/5-4_topk_permuted_letter_kept.sh global_facts

MODEL_NAME=Qwen3-8B MODEL_PATH=/path/to/Qwen3-8B \
bash scripts/5-5_topk_permuted_letter_removed.sh global_facts
```

五个条件的精确定义如下：

- `letter_kept`：不删除 A/B/C/D 标签，直接在完整候选中保留高 attention 内容。
- `letter_removed`：先删除 `A.`、`(B)`、`**C**` 等独立选项标签，再根据清洗后的候选数重新计算 k。`According`、`Data` 等普通词不会因为包含字母 A/B/C/D 而被删除。
- `correct_option_words_removed`：先执行 `letter_removed`，再从正确选项文本提取小写、Unicode/标点规范化且去除英文停词后的词和数字，在 CoT 与 explanation 中进行完整词匹配并删除，最后重新计算 k。
- `permuted_letter_kept`：使用与 `letter_kept` 相同的 retained CoT，不重新计算 attention；只重排选项文本并更新 prompt 和 truth。
- `permuted_letter_removed`：使用与 `letter_removed` 相同的 retained CoT，再执行相同的选项文本置换。

置换不是把标签名称改掉。A/B/C/D 仍按正常顺序显示，变化的是标签后面的选项文本。例如：

```text
原始：A. apple   B. banana   C. cherry   D. date，truth=B
置换：A. cherry  B. date     C. banana   D. apple，truth=C
```

prompt 中四个选项块和 truth 必须同步更新，retained CoT 保持逐字不变。每条样本使用 `Random(51 + sample_index)` 得到一次固定非原序置换，并保证正确选项移动到不同字母。选项块无法唯一定位时实验会立即失败。置换条件会在置换后的题目上额外重算一次 No-CoT 参照，从而保证置信度 delta 比较的是同一 prompt 和同一 truth。

新实验的候选范围包括完整 CoT 和 `</think>` 到 `\boxed` 之间的 explanation，明确关闭 v4 的“自动删除最后两句”逻辑。每个 ratio 使用 `round(candidate_count * ratio)`。记录中保存筛选前后候选数、目标/实际 k、全局位置、词、attention、segment、删除原因和实际保留比例。

置换与正确选项词清洗以 `input_text` 中模型实际看到的 A/B/C/D 选项块为准，同时将原始 `choices` 字段保存在 `source_choices_field`。这是为了兼容旧数据中选项本身含 `|` 或 `||`、但早期预处理按 `|` 切分而导致 prompt 与序列化字段不一致的情况；新实验不会静默把题目换成模型当时没有看到的版本。

Top-k 输出为：

```text
final_results/case_study/topk_leakage/{data_name}/{model_name}/
├── {subset}/{condition}/
│   ├── top_percentage0.1.json
│   ├── top_percentage0.1_good_cases.json
│   ├── top_percentage0.1_bad_cases.json
│   ├── top_percentage0.1_summary.json
│   └── 对应 0.2、0.3 文件
├── {condition}_aggregate_summary.json
└── {condition}_aggregate_summary.csv
```

普通 Top-k 条件会按 `sample_index` 读取已经生成的 No-CoT records，并保存 prediction、correct、truth probability 相对 No-CoT 的变化。若 No-CoT 文件不存在，实验仍可运行，但这些 delta 字段为 `null`，因此建议先运行 No-CoT baseline。

### 12.4 人工 CoT 干预

`15_manual_cot_intervention.py` 接收只读 JSON。每个 case 可以包含多个修改版本：

```json
[
  {
    "case_id": "global_facts_0001",
    "data_name": "mmlu_redux",
    "model_name": "Qwen3-8B",
    "subset": "global_facts",
    "sample_index": 0,
    "base_input_text": "完整原始题目 prompt",
    "truth": "C",
    "baseline": {
      "cot": "原始或保留后的 CoT",
      "explanation": "原始或保留后的 explanation"
    },
    "variants": [
      {
        "variant_id": "minimal_scrub",
        "cot": "人工删除答案线索后的 CoT",
        "explanation": "人工修改后的 explanation",
        "edit_note": "removed direct answer clues"
      },
      {
        "variant_id": "counterfactual",
        "cot": "人工反事实修改后的 CoT",
        "explanation": "人工修改后的 explanation",
        "edit_note": "changed answer-bearing number"
      }
    ]
  }
]
```

运行方式：

```bash
INPUT_FILE=path/to/manual_cases.json \
RUN_NAME=minimal_scrub_study \
MODEL_NAME=Qwen3-8B \
MODEL_PATH=/path/to/Qwen3-8B \
bash scripts/6_manual_cot_intervention.sh
```

baseline 和所有 variants 都会重新预测，不沿用旧 prediction。若旧记录缺少 `base_input_text`，可以提供 `input_text_with_CoT`；脚本只在其中恰好存在一个 `<think>` 时提取其前缀，否则明确报错。输入文件本身不会被修改。

输出位于：

```text
final_results/case_study/manual/{data_name}/{model_name}/{run_name}/
```

其中包含嵌套 `records.json`、baseline/variant 的 records 与 good/bad cases、`improved_cases.json`、`degraded_cases.json` 和 `summary.json`。`by_variant/{variant_id}/` 还会分别保存每个修改类型的 records、good、bad、improved、degraded 和 summary。summary 给出修改前后准确率、accuracy delta、平均 truth-probability delta、`wrong→correct`、`correct→wrong` 以及答案 changed/unchanged 数量。

### 12.5 通用启动参数和结果读取

启动脚本支持以下环境变量：

```text
DATA_NAME      数据集名（默认 mmlu_redux）
MODEL_NAME     结果文件中的模型名
MODEL_PATH     本地模型路径
OUTPUT_ROOT    输出根目录（默认 ./final_results/case_study）
CUDA_VISIBLE_DEVICES  可见 GPU（默认 0）
PYTHON_BIN     Python 命令（默认 python）
MAX_SAMPLES    可选的小样本 smoke test 数量
SKIP_EXISTING  No-CoT/Top-k 中设为 1 时跳过完整已有输出
SUBSETS        可选的空格分隔 subset；也可直接作为 shell 位置参数传入
```

建议分析时同时查看：

1. `records` 中的生成答案变化；
2. good/bad case 的样本级内容；
3. `truth_probability_normalized` 的变化，即使答案字母没有改变；
4. subset summary 和 aggregate summary 的 micro accuracy。

跨 subset 的 `micro accuracy` 按所有 subset 的正确样本总数除以总样本数计算，不是各 subset accuracy 的简单平均。

当前仓库已有本地产物主要来自 Qwen3。新脚本没有写死 Qwen3 模型名，并沿用项目统一 loader，因此 DeepSeek-R1 等模型可以使用相同 CLI；但在没有对应本地模型和产物的机器上，不能把接口兼容解释为已经完成实测。
