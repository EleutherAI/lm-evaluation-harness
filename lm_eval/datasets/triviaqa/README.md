---
dataset_info:
  features:
  - name: question_id
    dtype: string
  - name: question_source
    dtype: string
  - name: question
    dtype: string
  - name: answer
    struct:
    - name: aliases
      sequence: string
    - name: value
      dtype: string
  - name: search_results
    sequence:
    - name: description
      dtype: string
    - name: filename
      dtype: string
    - name: rank
      dtype: int32
    - name: title
      dtype: string
    - name: url
      dtype: string
    - name: search_context
      dtype: string
  config_name: triviaqa
  splits:
  - name: train
    num_bytes: 1270894387
    num_examples: 87622
  - name: validation
    num_bytes: 163755044
    num_examples: 11313
  download_size: 632549060
  dataset_size: 1434649431
---
