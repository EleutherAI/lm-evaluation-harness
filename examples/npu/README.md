# NPU Examples

This directory contains examples and demonstrations for the Rebellions NPU integration supporting **causal and seq2seq language models**.

## Files

- **`demo_working_rbln_integration.py`** - Complete working demonstration of the NPU integration

## Running Examples

```bash
# Run the working demo
python examples/npu/demo_working_rbln_integration.py
```

## Example Commands

### Currently Supported Models

The `rbln` model automatically detects the optimal RBLN Auto class for supported models:

#### Causal Language Models
```bash
# Text generation - LLaMA models (auto-detects RBLNAutoModelForCausalLM)
lm_eval --model rbln --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct --tasks hellaswag

# Language modeling - GPT-2 (auto-detects RBLNAutoModelForCausalLM)
lm_eval --model rbln --model_args pretrained=gpt2 --tasks lambada_openai

# Conversational AI - DialoGPT (auto-detects RBLNAutoModelForCausalLM)
lm_eval --model rbln --model_args pretrained=microsoft/DialoGPT-small --tasks hellaswag
```

#### Seq2Seq Models
```bash
# Text classification - T5 models (auto-detects RBLNAutoModelForSeq2SeqLM)
lm_eval --model rbln --model_args pretrained=t5-small --tasks cola,sst2

# Question answering - Flan-T5 models (auto-detects RBLNAutoModelForSeq2SeqLM)
lm_eval --model rbln --model_args pretrained=google/flan-t5-base --tasks squadv2

# Text understanding - BART models (auto-detects RBLNAutoModelForSeq2SeqLM)
lm_eval --model rbln --model_args pretrained=facebook/bart-base --tasks glue
```

#### Multi-NPU Configurations
```bash
# Multi-NPU tensor parallelism with large models
lm_eval --model rbln --model_args 'pretrained=meta-llama/Meta-Llama-3-8B-Instruct,rbln_tensor_parallel_size=4' --tasks hellaswag
```

### Unsupported Models (Coming Soon)

 The following model types will be added in future releases:
```bash
# BERT models - NOT SUPPORTED YET (will show clear error message)
# lm_eval --model rbln --model_args pretrained=bert-base-uncased --tasks cola

# Vision models - NOT SUPPORTED YET
# Audio models - NOT SUPPORTED YET
# Multimodal models - NOT SUPPORTED YET
```

For the complete list of RBLN-compatible models, see the [Rebellions Model Zoo](https://rebellions.ai/developers/model-zoo/).
