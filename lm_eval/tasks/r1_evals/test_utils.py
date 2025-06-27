import pandas as pd
from utils import postprocess,postprocess_target
df = pd.read_json("/root/.cache/huggingface/reasoning_evals/qwq_evals_final/Qwen__QwQ-32B/samples_demo_aime24_2025-04-03T19-38-22.380625.jsonl", lines=True,dtype={'target': str})
df["candidate"] = df["filtered_resps"].apply(lambda x: postprocess(x[0]))
# df["candidate"] = df["candidate"].apply(lambda x:  strip_string(x))
# df_target = pd.read_json("/vllm-workspace/reasoning_evals/unit_test.jsonl",lines=True,dtype={'answer': str})
df[["candidate","target","exact_match"]]
df["target_clean"] = df["target"].apply(lambda x: postprocess_target(x))
df[["candidate","target_clean","exact_match"]]

from utils import process_results_math
postprocess_target("00044")
match_list = []

for i,j in zip(df["filtered_resps"],df["target"]):
    doc= {"answer":j}
    match_list.append(process_results_math(doc,i))
match_list

df = pd.read_json("/root/.cache/huggingface/reasoning_evals/qwq_evals_final/Qwen__QwQ-32B/samples_demo_gpqa_diamond_2025-04-04T06-51-41.111378.jsonl", lines=True,dtype={'target': str})
df["candidate"] = df["filtered_resps"].apply(lambda x: postprocess(x[0]))
df["target_clean"] = df["target"].apply(lambda x: postprocess_target(x))
df[["candidate","target","exact_match"]]

from utils import process_results_gpqa

match_list = []

for i,j in zip(df["filtered_resps"],df["target"]):
    doc= {"answer":j}
    match_list.append(process_results_gpqa(doc,i))
    
match_list



import json 
data = []
with open("/vllm-workspace/reasoning_evals/unit_test.jsonl","r",encoding="utf-8") as f:
    for line in f:
        stripped_line = line.strip()
        if stripped_line:
            try:
                data.append(json.loads(stripped_line))
                print(data[-1])
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line due to JSON decoding error: {e}") 

res = df[23:27]["filtered_resps"]

for d,r in zip(data,res):
    d["response"] = r


with open("unittest.jsonl","w",encoding="utf-8") as f:
    for item in data:
        json_string = json.dumps(item,ensure_ascii=False)
        f.write(json_string+"\n")