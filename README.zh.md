# 语言模型评估框架 (Language Model Evaluation Harness)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10256836.svg)](https://doi.org/10.5281/zenodo.10256836)

---

## 最新动态 📣
- [2026/09] **插件机制 (Plugins)**：模型后端、过滤器、评估指标与聚合算法现已支持直接在您自己的软件包中注册，无需 Fork 官方仓库——声明一个 `lm_eval.*` 入口点即可实现零配置自动发现，或通过 `--plugins` 指向本地模块。详见[插件指南](./docs/plugins.md)。
- [2025/12] **CLI 全新重构**：引入子命令（`run`、`ls`、`validate`）并通过 `--config` 提供对 YAML 配置文件的支持。详见 [CLI 参考手册](./docs/interface.md) 与[配置指南](./docs/config_files.md)。
- [2025/12] **更轻量的安装体验**：基础包不再默认捆绑 `transformers`/`torch`。请按需单独安装所需模型后端：`pip install lm_eval[hf]`、`lm_eval[vllm]` 等。
- [2025/07] 为 `hf`（token/str）、`vllm` 和 `sglang`（str）新增 `think_end_token` 参数，用于在支持的模型中剥离思维链 (CoT) 推理轨迹。
- [2025/03] 新增对 Hugging Face 模型引导控制 (Steering) 的支持！
- [2025/02] 新增对 [SGLang](https://docs.sglang.ai/) 的支持！
- [2024/09] 原型功能：支持创建与评估图文多模态输入、文本输出的任务，并新增了 `hf-multimodal`、`vllm-vlm` 模型类型和 `mmmu` 任务作为原型特性。欢迎体验并进行压力测试；同时推荐探索 [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval)（源自 lm-evaluation-harness 的优秀项目），以获取更全面的多模态任务、模型与特性支持。
- [2024/07] [API 模型](docs/API_guide.md)支持已全面重构更新，引入了批处理与异步请求支持，大幅降低了自定义与扩展门槛。**若要评估 Llama 405B，推荐使用 vLLM 提供的兼容 OpenAI 协议的 API 进行模型托管，并使用 `local-completions` 模型类型进行评估。**
- [2024/07] 新增 Open LLM Leaderboard 相关任务！可在 [leaderboard](lm_eval/tasks/leaderboard/README.md) 任务组下查看。

---

## 版本公告

**lm-evaluation-harness v0.4.0 正式发布**！

全新更新与特性包括：

- **新增 Open LLM Leaderboard 相关任务！可在 [leaderboard](lm_eval/tasks/leaderboard/README.md) 任务组下查看。**
- 内部架构全面重构
- 基于配置文件的任务创建与参数设定
- 更轻松地导入与共享外部定义的任务 YAML 配置
- 支持 Jinja2 提示词模板设计，便捷修改 Prompt 并支持从 Promptsource 导入
- 更丰富的高级配置选项，包括输出后处理、答案抽取、单文档多次生成、灵活的 Few-shot 设定等
- 速度提升与新型模型库支持：包括更快的数据并行 HF 模型调用、vLLM 支持、HuggingFace 的 MPS 后端加速等
- 日志系统与易用性改进
- 新增评测任务，包括 CoT BIG-Bench-Hard、Belebele、用户自定义任务分组等

更多详细信息请参阅 `docs/` 目录下的官方文档页面。

项目开发将在 `main` 分支上持续进行，热烈欢迎各位通过 GitHub Issue、PR 或在 [EleutherAI Discord](https://discord.gg/eleutherai) 社区中提供反馈、提出新特性诉求与交流讨论！

---

## 项目概览

本项目提供了一个统一的基准测试框架，用于在海量不同的评估任务上对生成式大语言模型 (Generative Language Models) 进行全面评测。

**核心特性：**

- 涵盖 60 多个主流学术评测基准，已实现数百个子任务和评估变体。
- 支持通过 [transformers](https://github.com/huggingface/transformers/)（包括通过 [GPTQModel](https://github.com/ModelCloud/GPTQModel) 和 [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) 进行量化）、[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) 以及 [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/) 加载的模型，具备灵活且与 Tokenizer 解耦的接口抽象。
- 支持基于 [vLLM](https://github.com/vllm-project/vllm) 的高吞吐量、显存友好的高速推理加速。
- 支持包括 [OpenAI](https://openai.com) 和 [TextSynth](https://textsynth.com/) 在内的商业化 API 接口。
- 支持在 [HuggingFace PEFT 库](https://github.com/huggingface/peft) 支持的适配器（如 LoRA）上进行针对性评测。
- 支持本地自有模型与自定义评测基准。
- 采用公开透明的评估提示词 (Prompts)，确保各篇研究论文之间的可复现性与横向可比性。
- 便捷支持自定义 Prompt 模板与自定义评估度量指标。

Language Model Evaluation Harness 是 🤗 Hugging Face 知名排行榜 [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 的官方底层技术基准，已被[数百篇学术论文](https://scholar.google.com/scholar?oi=bibs&hl=en&authuser=2&cites=15052937328817631261,4097184744846514103,1520777361382155671,17476825572045927382,18443729326628441434,14801318227356878622,7890865700763267262,12854182577605049984,15641002901115500560,5104500764547628290) 引用，并在 NVIDIA、Cohere、BigScience、BigCode、Nous Research、Mosaic ML 等数十家前沿顶尖机构的内部评测管线中广泛采用。

## 安装指南

如需从 GitHub 源码仓库安装 `lm-eval`，请执行：

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### 安装模型后端依赖

基础安装仅包含核心评估框架。**模型后端需按需通过可选扩展 (Extras) 单独安装**：

若需使用 HuggingFace transformers 模型：

```bash
pip install "lm_eval[hf]"
```

若需使用 vLLM 推理加速后端：

```bash
pip install "lm_eval[vllm]"
```

若需使用基于 API 的模型（OpenAI、Anthropic 等）：

```bash
pip install "lm_eval[api]"
```

多个模型后端支持合并一并安装：

```bash
pip install "lm_eval[hf,vllm,api]"
```

本文档末尾提供了全部可选额外依赖的详细完整清单。

## 基础用法

### 核心文档索引

| 指南手册 | 详细说明 |
|---|---|
| [CLI 参考手册](./docs/interface.md) | 命令行参数与所有子命令完整参考 |
| [配置指南](./docs/config_files.md) | YAML 配置文件格式规范与范例 |
| [Python API 接口](./docs/python-api.md) | 基于 `simple_evaluate()` 的代码级编程式调用 |
| [评测任务指南](./lm_eval/tasks/README.md) | 支持的任务清单与任务配置方法 |

运行 `lm-eval -h` 可查看全局通用参数，或使用 `lm-eval run -h` 查看具体的评估运行选项。

列出当前支持的所有评估任务：

```bash
lm-eval ls tasks
```

### Hugging Face `transformers`

> [!Important]
> 若要使用 HuggingFace 后端，请先安装对应依赖：`pip install "lm_eval[hf]"`

若要在 `hellaswag` 任务上评估托管于 [HuggingFace Hub](https://huggingface.co/models) 上的模型（例如 GPT-J-6B），可以使用如下命令（假设当前环境配备兼容 CUDA 的 GPU）：

```bash
lm_eval --model hf     --model_args pretrained=EleutherAI/gpt-j-6B     --tasks hellaswag     --device cuda:0     --batch_size 8
```

可通过 `--model_args` 参数向模型构造函数传递额外参数。尤其值得注意的是，这支持了在 Hub 上利用 `revisions` 加载训练中间阶段 Checkpoint 的通用实践，或指定模型运行精度：

```bash
lm_eval --model hf     --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float"     --tasks lambada_openai,hellaswag     --device cuda:0     --batch_size 8
```

框架全面支持通过 HuggingFace 中的 `transformers.AutoModelForCausalLM`（自回归、Decoder-only 的 GPT 风格模型）以及 `transformers.AutoModelForSeq2SeqLM`（如 T5 等 Encoder-Decoder 模型）加载的模型。

将 `--batch_size` 设为 `auto` 可开启自动批大小检测。系统将自动探测当前显卡所能容纳的最大 Batch Size。对于样本最长长度与最短长度差异较大的评测任务，周期性地重新计算最大批大小有助于进一步加速推理。为此，可以在该参数后追加 `:N`，表示在评估过程中重新计算 `N` 次最大 Batch Size。例如若需重新探测 4 次，命令如下：

```bash
lm_eval --model hf     --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float"     --tasks lambada_openai,hellaswag     --device cuda:0     --batch_size auto:4
```

> [!Note]
> 与为 `transformers.AutoModel` 传入本地路径同理，您也可以通过 `--model_args pretrained=/path/to/model` 为 `lm_eval` 传入本地模型存储路径。

#### 评估 GGUF 格式模型

`lm-eval` 支持利用 Hugging Face (`hf`) 后端直接评估 GGUF 格式的模型。这使得您可以评估兼容 `transformers`、`AutoModel` 以及 llama.cpp 转换导出的量化模型。

评估 GGUF 模型时，请通过 `--model_args` 参数传入包含模型权重的目录路径、`gguf_file` 文件名，并可选指定一个独立的 `tokenizer` 路径。

**🚨 重要注意事项：**  
若未显式提供单独的分词器路径，Hugging Face 将尝试从 GGUF 文件中动态逆向重构分词器——该过程可能耗费**数小时**甚至出现卡死挂起。传入单独的 tokenizer 可彻底避免此问题，将分词器加载耗时从数小时缩减至数秒。

**✅ 推荐用法：**

```bash
lm_eval --model hf     --model_args pretrained=/path/to/gguf_folder,gguf_file=model-name.gguf,tokenizer=/path/to/tokenizer     --tasks hellaswag     --device cuda:0     --batch_size 8
```

> [!Tip]
> 请确保 tokenizer 路径指向一个合法的 Hugging Face 分词器目录（例如包含 `tokenizer_config.json`、`vocab.json` 等文件）。

#### 评估由 llama.cpp 托管的 GGUF 模型

`gguf` 模型类型支持直接评测由 [llama.cpp](https://github.com/ggml-org/llama.cpp) 服务端 (`llama-server`) 部署并通过兼容 OpenAI 的 `/v1/completions` 端点暴露的模型：

```bash
lm_eval --model gguf     --model_args base_url=http://127.0.0.1:8080     --tasks hellaswag
```

- 要求使用 2024 年 12 月或更新版本的 llama.cpp，新版本以现代 OpenAI 格式返回 logprob（`logprobs.content`，详见 [llama.cpp#10783](https://github.com/ggml-org/llama.cpp/pull/10783)）。不再支持已被弃用的旧版格式 (`token_logprobs`)。
- 若服务端运行在多模型路由模式下，请传入模型名称或别名：`--model_args base_url=http://127.0.0.1:8080,model=my-model-alias`。
- 请求采用并发异步发送。默认并发度将根据服务端的槽位数（Slot count）自动探测（通过 `/props` → `total_slots`，即 llama-server 的 `--parallel` 设置）；亦可通过 `parallel=<N>` 进行手动覆盖。`loglikelihood`（对数似然）请求会通过 llama.cpp 的 `id_slot` 参数固定绑定到特定槽位，共享上下文的连续请求（例如多项选择题的多个候选补全选项）将被分配至同一个槽位，以便复用已缓存的 Prompt 前缀。`generate_until` 请求则不进行槽位固定，因为服务端动态闲置槽位调度能更好地平衡变长生成请求的负载。
- `loglikelihood` 采用精确的教师强制（Teacher Forcing）机制实现：llama.cpp 默认忽略 `echo` 且从不返回输入提示词的 logprob，因此补全部分会通过服务端的 `/tokenize` 接口进行分词（上下文与补全内容分别编码，与 HF 后端逻辑严格对齐），每个补全 token 均以其自然前缀作为 token-id 提示词，并附加一个 `logit_bias` 强制服务端对其进行采样。llama.cpp 会报告该强制 token 在*采样前*的对数概率以及无偏的 `top_logprobs`，从而在量化底噪容差范围内产生与 HF 后端高度一致的对数似然结果。（一种朴素替代方案——即使用 GBNF 语法强制补全——**无法**正常工作：语法约束解码可能选择非自然的碎片化分词，导致其 logprob 偏离真实分词对数似然，系统性地拉低最终得分）。
- 该模型类型目前暂未实现 `loglikelihood_rolling`（针对 wikitext 等困惑度 PPL 评测任务）。

#### 基于 Hugging Face `accelerate` 的多 GPU 评估

我们支持通过 Hugging Face 的 [accelerate 🚀](https://github.com/huggingface/accelerate) 库进行多卡并行评估的三种核心模式。

若要执行**数据并行评估**（每个 GPU 加载一份**独立的完整模型副本**），我们可以直接使用 `accelerate` 启动器：

```bash
accelerate launch -m lm_eval --model hf     --tasks lambada_openai,arc_easy     --batch_size 16
```

（或使用 `accelerate launch --no-python lm_eval`）。

若您的单张显卡足以容纳单个模型实例，此方案可让您在 K 张 GPU 上以约 K 倍的速度完成评测。

**⚠️ 警告**：该数据并行方案与 FSDP 模型分片不兼容，因此在 `accelerate config` 中必须禁用 FSDP，或者选用 NO_SHARD FSDP 模式。

使用 `accelerate` 的第二种多 GPU 评估场景是**模型体量过大、单张显卡无法容纳**时。

在这种情况下，**无需通过 `accelerate` 启动器运行**，而是向 `--model_args` 传入 `parallelize=True`：

```bash
lm_eval --model hf     --tasks lambada_openai,arc_easy     --model_args parallelize=True     --batch_size 16
```

这会将模型的权重自动分片切分到所有可用的 GPU 上。

对于高级用户或更大规模的模型，在启用 `parallelize=True` 时还支持配置如下进阶参数：

- `device_map_option`：模型权重在可用 GPU 间的切分分配策略，默认为 `"auto"`。
- `max_memory_per_gpu`：加载模型时每张 GPU 允许使用的最大显存配额。
- `max_cpu_memory`：在将模型权重卸载到系统内存 (RAM) 时允许使用的最大 CPU 内存。
- `offload_folder`：在显存与内存均不足时，模型权重溢出卸载到本地磁盘的目标目录。

第三种方式是将两者结合使用。这将使您同时兼享数据并行与模型分片优势，对于体量极大且拥有充足多卡资源的场景尤为高效：

```bash
accelerate launch --multi_gpu --num_processes {nb_of_copies_of_your_model}     -m lm_eval --model hf     --tasks lambada_openai,arc_easy     --model_args parallelize=True     --batch_size 16
```

如需深入了解模型并行及其在 `accelerate` 库中的用法，请参阅 [accelerate 官方文档](https://huggingface.co/docs/transformers/v4.15.0/en/parallelism)。

**⚠️ 警告：`hf` 模型类型原生不直接支持多节点（Multi-node）分布式评估！关于如何编写多机评估脚本，请参考[我们的 GPT-NeoX 集成范例](https://github.com/EleutherAI/gpt-neox/blob/main/eval.py)。**

**说明：目前我们暂未原生提供多节点评估能力，建议使用托管推理服务对外提供接口，或针对您的分布式计算框架构建自定义适配层（例如 [GPT-NeoX 库的集成实现](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py)）。**

#### 张量并行（PyTorch 原生 Tensor Parallelism）

对于支持 PyTorch 原生张量并行（基于 DTensor）的模型，您可以通过在 `--model_args` 中设置 `tp_plan=auto`，在不依赖 `accelerate` device-map 的情况下跨 GPU 切分模型权重。使用 `torchrun` 或 `accelerate launch` 启动：

```bash
torchrun --nproc-per-node=4 -m lm_eval     --model hf     --model_args pretrained=google/gemma-4-31B-it,tp_plan=auto     --tasks lambada_openai,arc_easy     --batch_size 16
```

**使用约束：**

- `tp_plan` 与 `parallelize=True` 互斥——二者只能选其一。
- 模型的键值头数（Key-Value Heads）必须能被 `--nproc-per-node`（张量并行度 TP degree）整除。
- 需要 PyTorch >= 2.4 以及支持该模型 TP 规划的 `transformers` 版本（v4.47+）。

### 引导控制 Hugging Face 模型 (Steered Models)

要对应用了引导向量 (Steering Vectors) 的 Hugging Face `transformers` 模型进行评估，请将模型类型指定为 `steered`，并提供预定义引导向量的 PyTorch 文件路径，或指定一个 CSV 文件以定义如何从预训练的 `sparsify` 或 `sae_lens` 模型中提取引导向量（使用此方法需要安装相应的可选依赖）。

指定预定义的引导向量配置：

```python
import torch

steer_config = {
    "layers.3": {
        "steering_vector": torch.randn(1, 768),
        "bias": torch.randn(1, 768),
        "steering_coefficient": 1,
        "action": "add"
    },
}
torch.save(steer_config, "steer_config.pt")
```

指定基于稀疏自编码器的派生引导向量：

```python
import pandas as pd

pd.DataFrame({
    "loader": ["sparsify"],
    "action": ["add"],
    "sparse_model": ["EleutherAI/sae-pythia-70m-32k"],
    "hookpoint": ["layers.3"],
    "feature_index": [30],
    "steering_coefficient": [10.0],
}).to_csv("steer_config.csv", index=False)
```

加载引导控制配置运行评测框架：

```bash
lm_eval --model steered     --model_args pretrained=EleutherAI/pythia-160m,steer_path=steer_config.pt     --tasks lambada_openai,hellaswag     --device cuda:0     --batch_size 8
```

### NVIDIA `nemo` 框架模型

[NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo) 是专为从事大语言模型研究和 PyTorch 开发人员构建的生成式 AI 框架。

若要评估 `nemo` 模型，首先按照 [NeMo 官方文档](https://github.com/NVIDIA/NeMo?tab=readme-ov-file#installation) 完成安装。强烈建议使用 NVIDIA 官方 PyTorch 或 NeMo 容器镜像，尤其是在编译安装 Apex 或其他深层依赖遇到问题时（参见[最新发布的镜像版本](https://github.com/NVIDIA/NeMo/releases)）。同时请按照本文档中的[安装说明](#安装指南)安装 lm evaluation harness 库。

NeMo 模型可以通过 [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/models) 或在 [NVIDIA Hugging Face 组织主页](https://huggingface.co/nvidia) 下载获取。在 [NVIDIA NeMo 框架仓库](https://github.com/NVIDIA/NeMo/tree/main/scripts/nlp_language_modeling) 中提供了转换脚本，可将 llama、falcon、mixtral 或 mpt 等主流模型的 `hf` Checkpoint 转换为 `nemo` 格式。

在单张 GPU 上运行 `nemo` 模型：

```bash
lm_eval --model nemo_lm     --model_args path=<path_to_nemo_model>     --tasks hellaswag     --batch_size 32
```

建议提前解压 `.nemo` 模型包，以避免在 Docker 容器内运行时因动态解压导致磁盘空间溢出：

```bash
mkdir MY_MODEL
tar -xvf MY_MODEL.nemo -C MY_MODEL
```

#### NVIDIA `nemo` 模型多 GPU 评估

默认仅使用单张 GPU。但我们在单节点内支持数据复制或张量/流水线并行评估。

1) 若要启用数据复制并行，请将 `model_args` 中的 `devices` 设为欲运行的数据副本数量。例如在 8 张 GPU 上运行 8 个数据副本的命令为：

```bash
torchrun --nproc-per-node=8 --no-python lm_eval     --model nemo_lm     --model_args path=<path_to_nemo_model>,devices=8     --tasks hellaswag     --batch_size 32
```

2) 若要启用张量并行和/或流水线并行，请设置 `model_args` 中的 `tensor_model_parallel_size` 和/或 `pipeline_model_parallel_size`。此外，`devices` 的数值必须严格等于两者乘积。例如在单节点 4 张 GPU 上运行张量并行度为 2、流水线并行度为 2 的命令如下：

```bash
torchrun --nproc-per-node=4 --no-python lm_eval     --model nemo_lm     --model_args path=<path_to_nemo_model>,devices=4,tensor_model_parallel_size=2,pipeline_model_parallel_size=2     --tasks hellaswag     --batch_size 32
```

请注意，推荐使用 `torchrun --nproc-per-node=<设备数> --no-python` 替代直接运行 `python` 命令，以确保模型高效加载至各显卡中。这对于多卡加载大型 Checkpoint 尤其重要。

暂不支持：跨多节点评估以及数据副本与张量/流水线并行的复合混用。

### Megatron-LM 架构模型

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 是 NVIDIA 开源的大规模 Transformer 预训练与微调框架。本后端支持直接加载与评估 Megatron-LM 格式的原始 Checkpoint，免去格式转换开销。

**环境要求：**
- 需预先安装 Megatron-LM 或通过 `MEGATRON_PATH` 环境变量指定其代码路径
- 具备 CUDA 加速的 PyTorch 环境

**环境准备：**

```bash
# 设置指向 Megatron-LM 安装目录的环境变量
export MEGATRON_PATH=/path/to/Megatron-LM
```

**基础用法（单 GPU）：**

```bash
lm_eval --model megatron_lm     --model_args load=/path/to/checkpoint,tokenizer_type=HuggingFaceTokenizer,tokenizer_model=/path/to/tokenizer     --tasks hellaswag     --batch_size 1
```

**支持的权重格式：**
- 标准 Megatron Checkpoint (`model_optim_rng.pt`)
- 分布式 Checkpoint（`.distcp` 格式，系统会自动检测）

#### 并行模式一览

Megatron-LM 后端支持如下并行模式：

| 并行模式 | 参数配置 | 详细说明 |
|---|---|---|
| 单 GPU (Single GPU) | `devices=1`（默认） | 标准单卡评估 |
| 数据并行 (Data Parallelism) | `devices>1, TP=1` | 每个 GPU 持有完整模型副本，输入数据切分并行 |
| 张量并行 (Tensor Parallelism) | `TP == devices` | 模型各层权重横跨切分至多个 GPU |
| 专家并行 (Expert Parallelism) | `EP == devices, TP=1` | 面向 MoE 架构，将不同专家切分至各个 GPU |

> [!Note]
> - 目前暂不支持流水线并行 (PP > 1)。
> - 专家并行 (EP) 不能与张量并行 (TP) 混合启用。

**数据并行范例（4 张 GPU，每卡持有一份完整副本）：**

```bash
torchrun --nproc-per-node=4 -m lm_eval --model megatron_lm     --model_args load=/path/to/checkpoint,tokenizer_model=/path/to/tokenizer,devices=4     --tasks hellaswag
```

**张量并行范例（TP=2）：**

```bash
torchrun --nproc-per-node=2 -m lm_eval --model megatron_lm     --model_args load=/path/to/checkpoint,tokenizer_model=/path/to/tokenizer,devices=2,tensor_model_parallel_size=2     --tasks hellaswag
```

**MoE 专家并行范例（EP=4）：**

```bash
torchrun --nproc-per-node=4 -m lm_eval --model megatron_lm     --model_args load=/path/to/moe_checkpoint,tokenizer_model=/path/to/tokenizer,devices=4,expert_model_parallel_size=4     --tasks hellaswag
```

**通过 extra_args 传入额外 Megatron 参数：**

```bash
lm_eval --model megatron_lm     --model_args load=/path/to/checkpoint,tokenizer_model=/path/to/tokenizer,extra_args="--no-rope-fusion --trust-remote-code"     --tasks hellaswag
```

> [!Note]
> 默认已启用 `--use-checkpoint-args` 标志，系统会自动从 Checkpoint 中恢复模型架构参数。对于经由 Megatron-Bridge 转换的权重，这通常已包含全部必要的模型配置。

#### 基于 OpenVINO 模型的流水线并行评估

支持对 OpenVINO 导出的模型在评估过程中启用流水线并行。

要开启流水线并行，请在 `model_args` 中设置 `pipeline_parallel=True`。此外，必须将 `device` 设为形如 `HETERO:<GPU 序号1>,<GPU 序号2>` 的值（例如 `HETERO:GPU.1,GPU.0`）。启用 2 路流水线并行的命令示例如下：

```bash
lm_eval --model openvino     --tasks wikitext     --model_args pretrained=<path_to_ov_model>,pipeline_parallel=True     --device HETERO:GPU.1,GPU.0
```

### 基于 `vLLM` 的张量 + 数据并行极致推理加速

我们深度集成了 vLLM，可对各类[受支持的模型架构](https://docs.vllm.ai/en/latest/models/supported_models.html)实现极速推理，尤其在将超大规模模型切分至多张 GPU 时表现优异。支持单卡或多卡环境下的张量并行 (TP)、数据并行 (DP) 或两者混合模式：

```bash
lm_eval --model vllm     --model_args pretrained={model_name},tensor_parallel_size={GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={model_replicas}     --tasks lambada_openai     --batch_size auto
```

要使用 vLLM，请执行 `pip install "lm_eval[vllm]"`。有关完整受支持的 vLLM 参数配置，请参考[我们的 vLLM 集成源码](https://github.com/EleutherAI/lm-evaluation-harness/blob/e74ec966556253fbe3d8ecba9de675c77c075bce/lm_eval/models/vllm_causallms.py)与 vLLM 官方文档。

> [!Note]
> 当 `data_parallel_size>1` 时，每个模型副本将作为一个独立的 [Ray](https://github.com/ray-project/ray) Actor 分发，需要预先 `pip install ray`。每个 Actor 会独占 `tensor_parallel_size` 张 GPU（默认为 1）。

vLLM 的输出偶尔可能与 Hugging Face 存在细微差异。我们将 Hugging Face 视作黄金参考基准，并提供了一个[比对校验脚本](./scripts/model_comparator.py)，用于严格核对 vLLM 与 HF 输出的一致性。

> [!Tip]
> 为获取最高吞吐与性能，强烈建议尽可能在 vLLM 下指定 `--batch_size auto`，以充分释放其连续批处理 (Continuous Batching) 调度的威力！

> [!Tip]
> 通过 model args 向 vLLM 传入 `max_model_len=4096` 或其他合理的最大序列长度，可以在配合 auto batch size 时大幅提速并防止爆显存（OOM），例如 Mistral-7B-v0.1 默认的最大上下文长度高达 32k。

### 基于 `SGLang` 的张量 + 数据并行离线快速批处理

我们支持将 SGLang 作为高效的离线批处理推理后端。其 **[Fast Backend Runtime](https://docs.sglang.ai/index.html)** 通过高度优化的显存管理机制和并行处理技术提供极致性能。核心亮点包含张量并行、连续批处理以及广泛的低比特量化支持（FP8/INT4/AWQ/GPTQ）。

要使用 SGLang 作为评测后端，请**提前按照官方文档完成安装**：[SGLang 安装指南](https://docs.sglang.io/get_started/install.html#install-sglang)。

> [!Tip]
> 由于高性能注意力算子库 [`Flashinfer`](https://docs.flashinfer.ai/) 的独立安装特性，我们未将 `SGLang` 依赖打包进 [pyproject.toml](pyproject.toml)。请注意 `Flashinfer` 对 `torch` 的底层构建版本有特定兼容性要求。

SGLang 的服务参数与其他后端略有不同，详情请查阅 [SGLang 服务端参数说明](https://docs.sglang.io/advanced_features/server_arguments.html)。调用范例如下：

```bash
lm_eval --model sglang     --model_args pretrained={model_name},dp_size={data_parallel_size},tp_size={tensor_parallel_size},dtype=auto     --tasks gsm8k_cot     --batch_size auto
```

> [!Tip]
> 遭遇显存不足 (OOM) 报错时（尤其在多项选择题任务中），可尝试如下排查方案：
>
> 1. 改用手动指定的固定 `batch_size`，而非 `auto`。
> 2. 通过调低 `mem_fraction_static` 降低静态 KV Cache 显存池占比——例如在模型参数中添加 `--model_args pretrained=...,mem_fraction_static=0.7`。
> 3. 在多 GPU 环境下，调大张量并行尺寸 `tp_size`。

### ONNX Runtime GenAI

我们支持 **ONNX Runtime GenAI**，用于对由 [ONNX Runtime GenAI 模型构建器 (Model Builder)](https://onnxruntime.ai/docs/genai/howto/build-model.html) 导出的 ONNX 格式大语言模型进行全平台通用评估。与 [Windows ML](#windows-ml) 后端不同，该后端原生支持 Linux、macOS 和 Windows，并通过跨平台的 `og.Config` API 动态选用硬件执行提供程序 (Execution Provider, 包括 CPU、CUDA、DirectML、WebGPU 以及基于 VitisAI/RyzenAI 的 AMD NPU)。

安装该后端及与您的硬件平台相适配的 Execution Provider Wheel 包：

```bash
# CPU 环境
pip install "lm_eval[onnxruntime-genai]"
# 需使用 CUDA / DirectML 替代 CPU 包时（互斥）：
#   pip install onnxruntime-genai-cuda
#   pip install onnxruntime-genai-directml
```

在指定的 Execution Provider 上评估 Model Builder 导出的 ONNX 模型：

```bash
lm_eval --model onnxruntime-genai     --model_args pretrained=/path/to/model_builder_output,execution_provider=cuda     --tasks hellaswag     --batch_size 1
```

`pretrained` 路径需传入 Model Builder 输出目录（包含 `genai_config.json`、ONNX 计算图文件以及 HF 分词器）或其中的 `.onnx` 模型文件。`execution_provider` 默认为 `cpu`；对于硬件加速卡可传入 `cuda`、`dml`、`VitisAI` 等。额外的 Provider 选项可通过 `provider_options` 传递。

> [!Note]
> 支持的架构涵盖 Model Builder 支持的全部模型类型（Llama、Phi、Qwen、Gemma、Mistral、Granite、ChatGLM 等）。推理以 Batch Size 1 运行，每次运行绑定单一 Execution Provider。

### ONNX Runtime 原生会话后端

我们还支持将**相同**的 Model Builder 导出模型直接送入原生 `onnxruntime.InferenceSession` 运行，跳过 GenAI 循环。当您希望评估结果直接对齐部署环境的真实运行时表现，或者需要 `onnxruntime-genai` 暂未构建的 Execution Provider 时（尤其面向 AMD GPU 的 **ROCm** 和 **MIGraphX**），推荐使用此后端。

```bash
# CPU 环境
pip install "lm_eval[onnxruntime]"
# 需使用 CUDA / ROCm 替代 CPU 包时（互斥）：
#   pip install onnxruntime-gpu
#   pip install onnxruntime-rocm
```

```bash
lm_eval --model onnxruntime     --model_args pretrained=/path/to/model_builder_output,execution_provider=rocm     --tasks hellaswag,arc_easy,wikitext     --batch_size 1
```

`execution_provider` 既接受常用简写别名（`cpu`、`cuda`、`rocm`、`migraphx`、`dml`、`openvino`、`tensorrt`、`vitisai`、`webgpu`、`qnn`），也接受完整的 ONNX Runtime Provider 规范全称。与 `onnxruntime-genai` 后端一致，若不显式指定，系统会自动遵循导出目录中 `genai_config.json` 所声明的 Provider 设置。由于 Model Builder 计算图中包含某些第三方扩展算子，非 CPU Provider 会保留 `CPUExecutionProvider` 作为每节点的兜底后备。

由于两个 ONNX 后端共享同一套打分逻辑且最终在相同的 ORT 底层算子中执行，相同模型在相同 Provider 下给出的评估分数完全一致；仓库中的 `tests/models/test_onnxruntime_parity.py` 作为持续自动化对齐测试对此进行保证。

> [!Note]
> 该后端目前支持 `loglikelihood`、`multiple_choice` 以及 `loglikelihood_rolling` 任务（包括 hellaswag、arc、mmlu、wikitext 等）。对于 gsm8k 等自回归生成式任务，请针对同一模型目录使用 `--model onnxruntime-genai`。
>
> 安装的 `onnxruntime` 版本必须足够新，以便解析当前 Model Builder 版本导出的自定义算子 Schema。较老版本的运行时可能会直接拒绝加载计算图——例如 12 输入的 `GroupQueryAttention` 算子在 ONNX Runtime 1.22 上会加载失败。

### Windows ML

我们支持 **Windows ML**，在 Windows 平台上实现硬件加速推理评测。支持在 CPU、GPU 以及 **NPU (神经处理单元)** 设备上开展评测。

关于 Windows ML 的详细介绍，请参阅官方文档：  
https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview

如需使用 Windows ML，请安装相应依赖：

```bash
pip install wasdk-Microsoft.Windows.AI.MachineLearning[all] wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap onnxruntime-windowsml onnxruntime-genai-winml
```

在 Windows 系统的 NPU/GPU/CPU 上评测 ONNX Runtime GenAI 格式大模型：

```bash
lm_eval --model winml     --model_args pretrained=/path/to/onnx/model     --tasks mmlu     --batch_size 1
```

> [!Note]
> Windows ML 后端仅适配 ONNX Runtime GenAI 模型格式。针对 `transformers.js` 导出的模型无法运行。您可通过检查模型目录中是否存在 `genai_config.json` 来确认模型格式。

> [!Note]
> 要在目标设备上运行 ONNX Runtime GenAI 模型，**必须**将原始模型针对对应的芯片厂商与设备类型进行量化转换。针对其他硬件转换的模型通常无法跨硬件正常运行。有关模型转换指南，请参阅 [Microsoft AI Tool Kit](https://code.visualstudio.com/docs/intelligentapps/modelconversion)。

### 模型 API 与托管推理服务

> [!Important]
> 若要评估基于 API 的模型，请先安装扩展：`pip install "lm_eval[api]"`

本库全面支持评估通过各大商业化 API 托管的模型服务，同时也积极兼容各类高性能本地/自建私有推理服务端。

调用托管 API 模型示例：

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
lm_eval --model openai-completions     --model_args model=davinci-002     --tasks lambada_openai,hellaswag
```

我们同样支持对接任何实现了与 OpenAI Completions 或 ChatCompletions 接口标准兼容的本地私有推理服务：

```bash
lm_eval --model local-completions --tasks gsm8k --model_args model=facebook/opt-125m,base_url=http://{yourip}:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=16
```

请注意，对于外部托管的模型，与本地模型相关的配置项（如 `--device`）不应传入且不会生效。正如您可以通过 `--model_args` 为本地模型构造函数传递任意参数一样，对于托管 API，您同样可以用其向 API 客户端透传参数。支持的参数列表请查阅对应托管服务的官方接口文档。

| API 或推理服务端 | 支持状态 | `--model <xxx>` 名称 | 支持的模型范围 | 请求支持类型 |
|---|---|---|---|---|
| OpenAI Completions | :heavy_check_mark: | `openai-completions`, `local-completions` | 全部 OpenAI Completions API 模型 | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| OpenAI ChatCompletions | :heavy_check_mark: | `openai-chat-completions`, `local-chat-completions` | [全部 ChatCompletions API 模型](https://platform.openai.com/docs/guides/gpt) | `generate_until` (无 logprobs) |
| Anthropic | :heavy_check_mark: | `anthropic` | [支持的 Anthropic 引擎](https://docs.anthropic.com/claude/reference/selecting-a-model) | `generate_until` (无 logprobs) |
| Anthropic Chat | :heavy_check_mark: | `anthropic-chat`, `anthropic-chat-completions` | [支持的 Anthropic 引擎](https://docs.anthropic.com/claude/docs/models-overview) | `generate_until` (无 logprobs) |
| [LiteLLM](https://github.com/BerriAI/litellm) (聚合接入 100+ 模型供应商) | :heavy_check_mark: | `litellm`, `litellm-chat`, `litellm-chat-completions` | [LiteLLM 支持的所有服务商模型](https://docs.litellm.ai/docs/providers) | `generate_until` (无 logprobs) |
| Textsynth | :heavy_check_mark: | `textsynth` | [全部受支持的引擎](https://textsynth.com/documentation.html#engines) | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Cohere | [:hourglass: - 受阻于 Cohere API 缺陷](https://github.com/EleutherAI/lm-evaluation-harness/pull/395) | N/A | [全部 `cohere.generate()` 引擎](https://docs.cohere.com/docs/models) | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| [Llama.cpp](https://github.com/ggerganov/llama.cpp) (通过 [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)) | :heavy_check_mark: | `gguf`, `ggml` | [llama.cpp 支持的全部模型](https://github.com/ggerganov/llama.cpp) | `generate_until`, `loglikelihood`（困惑度评测暂未实现） |
| vLLM | :heavy_check_mark: | `vllm` | [大多数 HF 因果语言模型](https://docs.vllm.ai/en/latest/models/supported_models.html) | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Mamba | :heavy_check_mark: | `mamba_ssm` | [通过 `mamba_ssm` 运行的 Mamba 架构模型](https://huggingface.co/state-spaces) | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Huggingface Optimum (Causal LMs) | :heavy_check_mark: | `openvino` | 通过 HF Optimum 转换为 OpenVINO™ 中间表示 (IR) 格式的任意 Decoder-only AutoModelForCausalLM | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Huggingface Optimum-intel IPEX (Causal LMs) | :heavy_check_mark: | `ipex` | 任意 Decoder-only AutoModelForCausalLM | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Huggingface Optimum-habana (Causal LMs) | :heavy_check_mark: | `habana` | 任意 Decoder-only AutoModelForCausalLM | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| 基于 AWS Inf2 的 Neuron (Causal LMs) | :heavy_check_mark: | `neuronx` | 支持在 [huggingface-ami inferentia2 镜像](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2) 上运行的任意 Decoder-only AutoModelForCausalLM | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| NVIDIA NeMo | :heavy_check_mark: | `nemo_lm` | [全部支持的模型](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/core/core.html#nemo-models) | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| NVIDIA Megatron-LM | :heavy_check_mark: | `megatron_lm` | [Megatron-LM GPT 系列模型](https://github.com/NVIDIA/Megatron-LM)（标准及分布式权重） | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Watsonx.ai | :heavy_check_mark: | `watsonx_llm` | [支持的 Watsonx.ai 引擎](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx) | `generate_until`, `loglikelihood` |
| ONNX Runtime GenAI | :heavy_check_mark: | `onnxruntime-genai` | [GenAI 格式的 ONNX 模型](https://onnxruntime.ai/docs/genai/howto/build-model.html)（跨平台：CPU/CUDA/DirectML/NPU） | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| ONNX Runtime | :heavy_check_mark: | `onnxruntime` | 通过原生 `InferenceSession` 运行的 [GenAI 格式 ONNX 模型](https://onnxruntime.ai/docs/genai/howto/build-model.html)（扩展支持 ROCm/MIGraphX） | `loglikelihood`, `loglikelihood_rolling` |
| Windows ML | :heavy_check_mark: | `winml` | [GenAI 格式的 ONNX 模型](https://code.visualstudio.com/docs/intelligentapps/modelconversion) | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| [您的本地自建推理服务！](docs/API_guide.md) | :heavy_check_mark: | `local-completions` 或 `local-chat-completions` | 支持兼容 OpenAI API 协议的自建服务，易于拓展适配其他协议 | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |

不输出 Logits 或 Logprobs 的黑盒模型仅能评测 `generate_until`（自由生成）类型的任务；而本地开源模型或可返回输入 Prompt 对数概率的 API，则能运行全部任务类型：`generate_until`、`loglikelihood`、`loglikelihood_rolling` 以及 `multiple_choice`（多项选择）。

有关任务不同 `output_types` 与模型请求类型的详细说明，请参考[我们的接口文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md#interface)。

> [!Note]
> 为了在 Anthropic Claude 3、GPT-4 等闭源对话 API 上获得最佳评估表现，强烈建议先通过添加 `--limit 10` 观察少量样本输出，以验证生成式任务的答案抽取与判分逻辑是否正常生效。在评估 anthropic-chat-completions 时，在 `--model_args` 中传入 `system="<提示内容>"` 明确指示模型的输出格式将十分有用。

### 其他框架集成

许多知名开源库均已内置直接调用 lm-evaluation-harness 的脚本支持，包括 [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py)、[Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md) 以及 [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py)。

若要在您自己的库中创建自定义集成，请参考[外部库集成指南](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage)。

### 更多特色功能

> [!Note]
> 对于不适宜直接在本地执行评估的任务（例如执行不受信任代码的安全风险，或评估流程极度复杂），可使用 `--predict_only` 参数仅保存模型解码生成文本，以便后续开展事后离线评估。

如果您配备支持 Metal 的 Mac，可通过将 `--device cuda:0` 替换为 `--device mps`（需 PyTorch 2.1 或更高版本）来利用 MPS 后端进行加速。**请注意，PyTorch MPS 后端仍处于早期迭代阶段，可能存在正确性缺陷或不支持的算子。若观察到 MPS 上的模型指标存在异常，建议首先比对 `--device cpu` 与 `--device mps` 的前向推理输出是否严格一致。**

> [!Note]
> 您可以通过运行如下命令检查输入到语言模型中的完整 Prompt 样貌：
>
> ```bash
> python write_out.py >     --tasks <task1,task2,...> >     --num_fewshot 5 >     --num_examples 10 >     --output_base_path /path/to/output/folder
> ```
>
> 这将为每个选定任务输出一个格式化的纯文本文件。

若需在执行评估的同时对评测任务的数据集完整性进行自检，可添加 `--check_integrity` 参数：

```bash
lm_eval --model openai     --model_args engine=davinci-002     --tasks lambada_openai,hellaswag     --check_integrity
```

## 高级使用技巧

对于通过 HuggingFace `transformers` 库加载的模型，任何通过 `--model_args` 传入的参数都会直接透传到底层对应的构造函数中。这意味着任何能在 `AutoModel` 中配置的选项均可在本库中无缝使用。例如，您可以通过 `pretrained=` 传入本地路径，或者评估通过 [PEFT](https://github.com/huggingface/peft) 微调的模型——只需在评估基座模型的调用命令中为 `model_args` 追加 `,peft=PATH`：

```bash
lm_eval --model hf     --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq     --device cuda:0
```

对于以增量权重 (Delta Weights) 形式发布的模型，同样可通过 Hugging Face transformers 轻松加载。在 `--model_args` 中，设置 `delta` 参数指定增量权重，并使用 `pretrained` 参数指定其作用的相对基座模型：

```bash
lm_eval --model hf     --model_args pretrained=Ejafa/llama_7B,delta=lmsys/vicuna-7b-delta-v1.1     --tasks hellaswag
```

评估 GPTQ 量化模型时，支持使用 [GPTQModel](https://github.com/ModelCloud/GPTQModel)（速度更快）或 [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)。

使用 GPTQModel：向 `model_args` 中追加 `,gptqmodel=True`：

```bash
lm_eval --model hf     --model_args pretrained=model-name-or-path,gptqmodel=True     --tasks hellaswag
```

使用 AutoGPTQ：向 `model_args` 中追加 `,autogptq=True`：

```bash
lm_eval --model hf     --model_args pretrained=model-name-or-path,autogptq=model.safetensors,gptq_use_triton=True     --tasks hellaswag
```

任务名称支持通配符过滤，例如可以通过 `--task lambada_openai_mt_*` 一键运行所有经机器翻译的 lambada 子任务。

## 评估结果保存与缓存

通过指定 `--output_path` 可保存完整的评估结果。系统还支持添加 `--log_samples` 参数将模型的具体输入输出样本持久化记录，以便事后深度分析。

> [!TIP]
> 使用 `--use_cache <DIR>` 可以缓存评估结果；在恢复同一组（模型，任务）的运行时，将自动跳过已评估过的样本。请注意缓存与显卡 Rank 绑定，因此中断后恢复时请保持相同的 GPU 数量。此外，还可以使用 `--cache_requests` 缓存数据集预处理步骤，大幅加快后续评估的启动速度。

若需将评估结果和采样输出直接推送到 Hugging Face Hub，请先确保已在 `HF_TOKEN` 环境变量中配置具备写入权限的 Access Token。随后使用 `--hf_hub_log_args` 参数指定组织名、仓库名、公开可见性以及是否推送结果与样本——[HF Hub 示例数据集](https://huggingface.co/datasets/KonradSzafer/lm-eval-results-demo)。示例如下：

```bash
lm_eval --model hf     --model_args pretrained=model-name-or-path,autogptq=model.safetensors,gptq_use_triton=True     --tasks hellaswag     --log_samples     --output_path results     --hf_hub_log_args hub_results_org=EleutherAI,hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False
```

推送完成后，您可以在 Python 中通过如下代码便捷下载历史评测结果与生成样本：

```python
from datasets import load_dataset

load_dataset("EleutherAI/lm-eval-results-private", "hellaswag", "latest")
```

有关支持参数的完整参考，请查阅官方文档中的[命令行与接口指南](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)！

## 结果可视化分析

您可以无缝集成 Weights & Biases (W&B) 和 Zeno，对评估结果进行多维度可视化与深度洞察。

### Zeno

您可以使用 [Zeno](https://zenoml.com) 探索与可视化您的评测输出。

首先前往 [hub.zenoml.com](https://hub.zenoml.com) 注册账号，并在[个人账户中心](https://hub.zenoml.com/account)获取 API Key。将其保存为环境变量：

```bash
export ZENO_API_KEY=[your api key]
```

同时需要安装扩展依赖：`pip install "lm_eval[zeno]"`。

运行评测时，需附带 `log_samples` 和 `output_path` 参数。系统期望 `output_path` 目录下包含代表各个独立模型的子目录。您可以对任意数量的任务和模型执行评测，并将全部结果上传为 Zeno 项目：

```bash
lm_eval     --model hf     --model_args pretrained=EleutherAI/gpt-j-6B     --tasks hellaswag     --device cuda:0     --batch_size 8     --log_samples     --output_path output/gpt-j-6B
```

随后通过 `zeno_visualize` 脚本将数据上传：

```bash
python scripts/zeno_visualize.py     --data_path output     --project_name "Eleuther Project"
```

该脚本会将 `data_path` 下的所有子文件夹识别为不同模型，并将各模型目录内的任务上传至 Zeno。若同时评测了多个任务，`project_name` 将作为项目前缀，并为每个评测任务单独创建一个可视化项目。

此工作流的完整代码示例请参考 [examples/visualize-zeno.ipynb](examples/visualize-zeno.ipynb)。

### Weights & Biases (W&B)

通过 [Weights & Biases](https://wandb.ai/site) 深度集成，您可以更加高效地分析与挖掘评测结果背后的规律。该集成旨在全面自动化使用 W&B 记录与可视化实验指标的流程。

集成具备如下能力：

- 自动化记录评测指标与结果汇总；
- 将采样样本沉淀为 W&B Tables，便于图形化交互对比；
- 将 `results.json` 沉淀为 Artifact 进行版本受控追踪；
- 记录 `<task_name>_eval_samples.json` 样本文件（启用 sample 记录时）；
- 一键生成包含全部关键度量指标的完整分析报表与仪表盘；
- 自动记录任务级与 CLI 运行配置；
- 开箱即用地记录评测运行命令、GPU/CPU 资源规格、执行时间戳等环境元数据。

首先安装 wandb 扩展：`pip install "lm_eval[wandb]"`。

在命令行中完成 W&B 认证登录：访问 https://wandb.ai/authorize 获取认证 Token，随后在终端执行 `wandb login`。

照常运行评测框架，并附带 `wandb_args` 参数。通过该参数以逗号分隔形式向 wandb 初始化过程 ([wandb.init](https://docs.wandb.ai/ref/python/init)) 传递参数：

```bash
lm_eval     --model hf     --model_args pretrained=microsoft/phi-2,trust_remote_code=True     --tasks hellaswag,mmlu_abstract_algebra     --device cuda:0     --batch_size 8     --output_path output/phi-2     --limit 10     --wandb_args project=lm-eval-harness-integration     --log_samples
```

在终端标准输出中，您将看到通往 W&B Run 控制台及自动生成实验报告的链接。完整工作流及超出 CLI 范围的代码级定制示例请参阅 [examples/visualize-wandb.ipynb](examples/visualize-wandb.ipynb)。

## 参与贡献

欢迎随时查阅我们的 [GitHub Issues 讨论区](https://github.com/EleutherAI/lm-evaluation-harness/issues) 并提交 Pull Request！

有关库内部架构及各模块协同逻辑的更多信息，请浏览[官方文档主页](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)。

如需开展本地开发，请先克隆仓库并安装开发环境依赖：

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[dev,hf]"
```

### 实现新评测任务

如需在评测框架中实现新任务，请参考[新增任务指南](./docs/new_task_guide.md)。

通常情况下，在处理 Prompt 设定及其他评测细节争议时，我们遵循如下优先级决策原则：

1. 若大语言模型训练研究人员已有广泛共识，遵循该公认流程。
2. 若存在清晰明确的官方基准实现，遵循官方实现。
3. 若大语言模型评测学术界已有广泛共识，遵循该公认流程。
4. 若存在多种常见实现但缺乏统一共识，在常见实现中选择我们推荐的方案。同上，优先从 LLM 训练论文中采用的方案中选取。

以上为指导原则而非死板教条，在特殊场景下可酌情权衡。

我们始终努力优先与其他主流评测体系保持一致，以减少研究人员跨论文对比模型得分时产生的潜在偏差。在历史上，我们也优先对齐了 [Language Models are Few Shot Learners](https://arxiv.org/abs/2005.14165) 论文中的具体实现，因为本框架的最初目标正是精确复现并比对该论文中的结论。

### 社区与技术支持

获取技术支持的最佳途径是在本仓库中创建 Issue，或加入 [EleutherAI Discord 服务器](https://discord.gg/eleutherai)。其中 `#lm-thunderdome` 频道专注于本项目的核心开发讨论，`#release-discussion` 频道则负责提供版本发布支持。无论您在使用过程中获得了良好体验还是遇到了困难，我们都非常期待听到您的声音！

## 可选扩展依赖表 (Optional Extras)

各类扩展依赖均可通过 `pip install -e ".[NAME]"` 按需安装：

### 模型后端 (Model Backends)

以下扩展用于安装运行特定模型后端所需的依赖项：

| 名称 (NAME) | 详细说明 |
|---|---|
| hf | HuggingFace Transformers（包含 torch、transformers、accelerate、peft） |
| vllm | vLLM 高速推理引擎 |
| api | API 模型（OpenAI、Anthropic、本地自建服务端） |
| gptq | AutoGPTQ 量化模型推理 |
| gptqmodel | GPTQModel 量化模型推理 |
| ibm_watsonx_ai | IBM watsonx.ai 模型 |
| ipex | Intel IPEX 加速后端 |
| habana | Intel Gaudi 加速后端 |
| optimum | Intel OpenVINO 模型支持 |
| neuronx | AWS Inferentia2 云实例支持 |
| onnxruntime-genai | ONNX Runtime GenAI（全平台） - CPU/CUDA/DirectML/NPU |
| onnxruntime | ONNX Runtime 原生会话 - 扩展支持 ROCm/MIGraphX |
| winml | Windows ML (ONNX Runtime GenAI) - CPU/GPU/NPU |
| sparsify | Sparsify 模型引导控制 |
| sae_lens | SAELens 模型引导控制 |

### 评测任务依赖 (Task Dependencies)

以下扩展用于安装特定评测任务所必需的数据集与解析工具：

| 名称 (NAME) | 详细说明 |
|---|---|
| tasks | 全部任务专用依赖总和 |
| acpbench | ACP Bench 评测任务 |
| audiolm_qwen | 通义千问 Qwen2 音频模型评测 |
| ifeval | IFEval 指令遵循评测任务 |
| japanese_leaderboard | 日语大语言模型排行榜任务 |
| longbench | LongBench 长文本评测任务 |
| math | 数学题解准确性校验 |
| multilingual | 多语言分词器 |
| ruler | RULER 超长上下文评测基准 |

### 开发与实用工具 (Development & Utilities)

| 名称 (NAME) | 详细说明 |
|---|---|
| dev | 代码规范检查 (Linting) 与贡献者开发套件 |
| hf_transfer | 加速 Hugging Face Hub 资源下载 |
| sentencepiece | Sentencepiece 分词组件 |
| unitxt | Unitxt 评估套件 |
| wandb | Weights & Biases 实验记录集成 |
| zeno | Zeno 评测结果可视化工具 |

## 引用本项目

```text
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
