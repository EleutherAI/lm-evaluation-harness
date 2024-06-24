CMD = """lm_eval --model hf \
    --model_args pretrained={model} \
   --tasks halftruthdetection \
    --device cuda:0 \
    --batch_size 8
"""
yaml_path = "/usr/src/app/lm_eval/tasks/my_dataset/halftruthdetection.yaml"
models = {
    "llama3": "meta-llama/Meta-Llama-3-8B" ,
    "mistrl-v2-base": "mistral-community/Mistral-7B-v0.2",
    "gemma": "google/gemma-7b",
    "mistralinstruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemmainstruct": "google/gemma-7b-it"
}
prompts = [
    """Given claim and evidence, predict if the claim is true, mostly-true, half-true, mostly-false, false.\nclaim:{{claim}}\nevidence:{{evidence}}\nlabel: """,
    """Given evidence, decide if the given claim is true, mostly-true, half-true, mostly-false, false.\nclaim:{{claim}}\nevidence:{{evidence}}\nlabel: """,
    """Given claim and evidence, find if the claim is true, mostly-true, half-true, mostly-false, or false.\nclaim:{{claim}}\nevidence:{{evidence}}\nlabel: """,
    """Identify if the claim is true, mostly-true, half-true, mostly-false, or false based on the evidence.\nclaim:{{claim}}\nevidence:{{evidence}}\nlabel: """,
    """Given claim and evidence, find if the claim is true, mostly-true, half-true, mostly-false, or false. 
```Label Description``` 
- \"true\": The claim is accurate and includes all relevant information. There are no omissions or distortions that could mislead the audience. 
- \"mostly-true\": The claim is accurate, but it might benefit from additional context to provide a complete picture. However, the absence of this context does not alter the claim's accuracy. 
- \"half-true\": The claim is true in a limited context. However, it omits crucial information that could significantly alter its interpretation, leading to potential misunderstanding or misinterpretation.  
- \"mostly-false\": The claim contains some elements of truth but distorts or misrepresents critical facts. Important information is omitted, which could lead to a misleading impression despite some truthful elements. 
- \"false\": The claim is inaccurate and contradicts established facts. The claim has no truth, and it is likely to mislead those who encounter it. 
```Input``` claim:{{claim}}\nevidence:{{evidence}}\nlabel: """,
]
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B \
    --tasks halftruthdetection \
    --device cuda:0 \
    --apply_chat_template \
    --batch_size 8
    
lm_eval --model hf \--model_args pretrained=meta-google/gemma-7b-it \--tasks halftruthdetection \--device cuda:2 \--batch_size 8 > output/gemma/instruct/p1.txt 2>&1