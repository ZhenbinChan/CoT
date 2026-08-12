DEFAULT_SUBSETS=(
  global_facts
  high_school_mathematics
  college_mathematics
  high_school_computer_science
  college_computer_science
  high_school_biology
  college_biology
  professional_law
  college_physics
  machine_learning
  sociology
  us_foreign_policy
  computer_security
  conceptual_physics
  econometrics
  business_ethics
  clinical_knowledge
  electrical_engineering
  elementary_mathematics
  formal_logic
)

for subset in "${SUBSETS[@]}"; do
  CUDA_VISIABLE_DEVICES=1 python 4_maintain_only_topk_words.py \
    --data_name mmlu_redux  \
    --sub_set "${subset}"  \
    --model_name Qwen3-8B  \
    --model_path /share/nlp/chenzhenbin/Workspaces/LLMs/Qwen3-8B
done

