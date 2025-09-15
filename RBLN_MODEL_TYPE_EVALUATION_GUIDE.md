# RBLN Model Type Evaluation Guide

## **Overview**

The Rebellions NPU integration provides **streamlined evaluation** for causal and seq2seq language models, ensuring that evaluation tasks match the capabilities and intended use of each supported model architecture.

---

## **Current Implementation (v1.0)**

### **Supported Models**
- **Causal LM Models**: GPT, LLaMA, Mistral, Qwen, Phi, etc.
- **Seq2Seq Models**: T5, BART, Pegasus, Marian, etc.

### **What We Fixed**
- **Per-Token Forward Pass**: Handles causal models that return only last-token logits
- **Cache Position Fixes**: Resolves RBLN SDK seq2seq forward pass issues
- **Proper Error Handling**: Clear messages for unsupported model types
- **Focused Architecture**: Clean codebase supporting core language model types

---

## **Supported Model Types & Recommended Tasks**

### **1. Causal LM Models** 
**Examples**: LLaMA, GPT, Mistral, Qwen, Phi, Falcon, BLOOM, OPT  
**Best For**: Autoregressive language modeling  
**Evaluation Support**: 
- **Loglikelihood tasks** (with per-token forward pass)
- **Generation tasks**

**Technical Details**:
- Automatically detects when RBLN model returns only last-token logits
- Performs individual forward passes for each continuation token
- Reconstructs full sequence logits for accurate loglikelihood calculation

**Recommended Tasks**:
- **Primary**: `hellaswag`, `lambada_openai`, `arc_easy`, `arc_challenge`, `winogrande`
- **Alternative**: `gsm8k`, `truthfulqa_mc1`, `truthfulqa_mc2`, `mmlu`

**Example**:
```bash
lm_eval --model rbln --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct --tasks hellaswag --limit 10
lm_eval --model rbln --model_args pretrained=gpt2 --tasks lambada_openai
```

---

### **2. Seq2Seq Models** 
**Examples**: T5, BART, Pegasus, Marian, mBART, BlenderBot  
**Best For**: Encoder-decoder tasks  
**Evaluation Support**: 
- **Loglikelihood tasks** (with cache_position fixes)
- **Generation tasks**

**Technical Details**:
- Applies cache_position scalar conversion fix for RBLN SDK compatibility
- Handles explicit decoder_input_ids and decoder_attention_mask generation
- Proper float32 dtype handling for attention masks

**Recommended Tasks**:
- **Primary**: `cola`, `sst2`, `mrpc`, `qqp`, `mnli`, `qnli`, `rte`, `wnli`
- **Alternative**: `gsm8k`, `drop`, `squadv2`, `glue`

**Example**:
```bash
lm_eval --model rbln --model_args pretrained=t5-small --tasks cola --limit 10
lm_eval --model rbln --model_args pretrained=google/flan-t5-base --tasks squadv2
```

---

### **3. Unsupported Models (Coming Soon)**

The following model types will be added in future releases:

**Masked LM Models**: BERT, RoBERTa, DistilBERT
- **Planned Support**: Classification and span-based tasks
- **Example**: `bert-base-uncased` for `cola`, `sst2` tasks

**Vision Models**: Vision Transformers, Image Classification models
- **Planned Support**: Image understanding tasks
- **Example**: `google/vit-base-patch16-224` for `imagenet` tasks

**Audio Models**: Wav2Vec, Whisper, Audio Classification models
- **Planned Support**: Audio understanding tasks
- **Example**: `facebook/wav2vec2-base` for `speech_commands` tasks

**Question Answering Models**: BERT-for-QA, RoBERTa-for-QA
- **Planned Support**: Extractive question answering
- **Example**: QA models for `squad`, `squadv2` tasks

**Current Behavior for Unsupported Models**:
```bash
# Attempting to use unsupported models shows clear guidance:
lm_eval --model rbln --model_args pretrained=bert-base-uncased --tasks cola
# Output:
# WARNING: Model type 'bert' is not supported.
# WARNING: Only causal and seq2seq models are supported.
# ERROR: Only 'causal' and 'seq2seq' models are currently supported.
```

---

## **Technical Implementation**

### **Model Registration**
```python
@register_model("rbln")
class RBLNLM(TemplateLM):
    # Single model class handles both causal and seq2seq
```

### **Smart Model Detection**
```python
def _get_rbln_model_class(self, pretrained: str):
    # Auto-detects RBLNAutoModelForCausalLM or RBLNAutoModelForSeq2SeqLM
    if self.model_type == "seq2seq":
        return RBLNAutoModelForSeq2SeqLM
    else:  # Default to causal
        return RBLNAutoModelForCausalLM
```

### **Per-Token Forward Pass (Causal Models)**
```python
# Detects if RBLN model returns only last-token logits
if logits.shape[1] == 1 and continuation_length > 1:
    # Perform individual forward passes for each token
    for token_idx in range(continuation_length):
        # Reconstruct full sequence logits
```

### **Cache Position Fix (Seq2Seq Models)**
```python
def _apply_cache_position_fix(self):
    # Converts None/Tensor cache_position to scalar for RBLN SDK
    # Handles torch.full() compatibility issues
```

### **Error Handling for Unsupported Types**
```python
if self.model_type not in ["causal", "seq2seq"]:
    raise ValueError(
        f"Only 'causal' and 'seq2seq' models are currently supported."
    )
```

---

## **Verification Results**

### **Causal Models** 
- **Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Task**: `hellaswag` (loglikelihood)
- **Result**: Success with per-token forward pass
- **Shape issue**: Fixed (no more shape mismatches)
- **Command**: `lm_eval --model rbln --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct --tasks hellaswag`

### **Seq2Seq Models**
- **Model**: `t5-small`  
- **Task**: `cola` (loglikelihood)
- **Result**: Success with cache_position fixes
- **torch.full() error**: Fixed with scalar conversion
- **Command**: `lm_eval --model rbln --model_args pretrained=t5-small --tasks cola`

### **Error Handling**
- **Unsupported models**: Clear error messages with guidance
- **Model type detection**: Automatic causal/seq2seq classification
- **Future expansion**: Ready for additional model types

---

## **Best Practices**

### **1. Choose Appropriate Tasks**
- **Causal Models**: Use loglikelihood tasks (`hellaswag`, `lambada_openai`, `arc_easy`, etc.)
- **Seq2Seq Models**: Use classification/generation tasks (`cola`, `sst2`, `squadv2`, etc.)

### **2. Follow Model Type Guidelines**
- Use the recommended tasks listed in this guide for each model type
- Start with primary tasks, then try alternatives if needed
- Pay attention to error messages for unsupported model types

### **3. Use Appropriate Limits**
- Start with `--limit 10` for testing
- Remove `--limit` for real evaluation
- Monitor performance and adjust batch sizes if needed

### **4. Multi-NPU Usage**
- Use `rbln_tensor_parallel_size` for large models
- Example: `--model_args 'pretrained=meta-llama/Meta-Llama-3-8B-Instruct,rbln_tensor_parallel_size=4'`

---

## **Summary**

The RBLN integration provides **streamlined evaluation** for core language model types:

- **Causal Models**: Full loglikelihood support with per-token forward pass fixes
- **Seq2Seq Models**: Complete support with cache_position scalar conversion fixes
- **Error Handling**: Clear guidance for unsupported model types
- **Future Ready**: Architecture supports easy addition of new model types
- **Performance**: All shape issues resolved with RBLN-specific optimizations

**Result**: Causal and seq2seq models can be evaluated with tasks that properly measure their intended capabilities, leading to meaningful accuracy results. Other model types will be added in future releases.
