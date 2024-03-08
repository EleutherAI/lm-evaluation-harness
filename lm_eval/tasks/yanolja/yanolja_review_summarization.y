task: yanolja_review_ko
dataset_name: yanolja_review_ko
dataset_path: json
dataset_kwargs:
  data_files: /data/shared/datasets/yanolja_evaluations/yanolja_review_summarization.jsonl
group: yanolja_summarization
output_type: generate_until
test_split: train
doc_to_text: "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful and concise answers to the user's questions.
Human: 주어진 한국어 text를 요약해주세요.
text: {{text}}
Assistant: "
doc_to_target: "" # Reference-free metric

metric_list:
  - metric: xcomet
    aggregation: mean
    higher_is_better: true

generation_kwargs:
  until:
    - "\n\n"
    - "\n"
    - "<|im_end|>"
    - "</s>"
metadata:
  version: 1.0
