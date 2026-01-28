# log_samples 输出格式说明

当设置 `log_samples: true` 时，lm-evaluation-harness 会生成两个文件：
1. `results_*.json` - 汇总的评估结果
2. `samples_*.jsonl` - 每个样本的详细输出（JSONL 格式，每行一个 JSON 对象）

## 文件结构

### 1. results_*.json（汇总结果）

包含任务级别的汇总指标和配置信息：

```json
{
  "results": {
    "task_name": {
      "alias": "task_name",
      "metric_name,filter_name": 0.85,
      "metric_name_stderr,filter_name": 0.02,
      ...
    }
  },
  "configs": {
    "task_name": {
      "task": "task_name",
      "dataset_path": "...",
      "doc_to_text": "...",
      "generation_kwargs": {...},
      ...
    }
  },
  "versions": {...},
  "n-shot": {...}
}
```

### 2. samples_*.jsonl（详细样本）

每行是一个 JSON 对象，包含单个样本的完整信息。

## samples_*.jsonl 格式详解

每个样本（JSON 对象）包含以下字段：

### 核心字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `doc_id` | int | 样本在数据集中的索引 |
| `doc` | dict | 原始数据样本（包含问题、答案等） |
| `target` | str | 目标答案（经过 `doc_to_target` 处理） |
| `arguments` | dict | 模型输入的参数（提示词和生成参数） |
| `resps` | list | 模型的原始响应（所有重复生成的响应） |
| `filtered_resps` | list | 经过过滤器处理后的响应 |
| `filter` | str | 使用的过滤器名称 |
| `metrics` | list | 该样本计算的所有指标名称 |
| `doc_hash` | str | 文档内容的哈希值 |
| `prompt_hash` | str | 提示词的哈希值 |
| `target_hash` | str | 目标答案的哈希值 |

### 指标字段

根据任务配置的 `metric_list`，每个样本还会包含相应的指标值：

- `exact_match`: 精确匹配分数（0.0 或 1.0）
- `f1`: F1 分数（如果有）
- 其他自定义指标...

### 详细字段说明

#### 1. `doc` - 原始数据样本

```json
{
  "question": "问题文本",
  "answer": "答案文本（如果数据集包含）",
  ...
}
```

包含数据集中该样本的所有原始字段。

#### 2. `arguments` - 模型输入参数

格式化的字典结构，便于序列化：

```json
{
  "gen_args_0": {
    "arg_0": "完整的提示词（包含 few-shot examples）",
    "arg_1": {
      "until": ["Q:", "\n\n"],
      "do_sample": true,
      "temperature": 0.7,
      "top_p": 0.95,
      "max_new_tokens": 1024
    }
  }
}
```

- `arg_0`: 完整的提示词字符串（包含 few-shot examples 和当前问题）
- `arg_1`: 生成参数（`generation_kwargs`）

#### 3. `resps` - 原始响应

嵌套列表结构：
- 外层列表：每个元素对应一个请求（通常只有一个请求，所以长度为 1）
- 内层列表：该请求的所有重复响应（长度为 `repeats` 参数值）

```json
[
  [
    "response_1",  // 第1次生成的响应
    "response_2",  // 第2次生成的响应
    ...
    "response_64"  // 第64次生成的响应（如果 repeats=64）
  ]
]
```

对于 `repeats=64` 的任务：
- `resps[0]` 包含 64 个字符串，每个字符串是模型的一次完整生成输出
- 每个响应都是完整的生成文本（可能包含思考过程、答案等）

#### 4. `filtered_resps` - 过滤后的响应

**含义**：经过过滤器（filter）处理后的响应列表。过滤器会对原始的 `resps` 进行后处理，例如：
- 使用正则表达式从文本中提取答案
- 进行多数投票（majority voting）
- 选择前 k 个响应等

**结构**：列表，每个元素对应一个请求（通常只有一个请求）

```json
["11"]  // 从原始响应中提取的答案
```

**处理流程示例**（以 gsm8k 任务的 `score-first` 过滤器为例）：

1. **原始响应** (`resps[0][0]`)：

   ```
   " Janet lays 16 eggs per day. She eats 3 eggs for breakfast...
    The answer is 11."
   ```

2. **应用正则过滤器** (`regex`):
   - 使用模式 `"The answer is (\\-?[0-9\\.\\,]*[0-9]+)"`
   - 从文本中提取数字：`"11"`

3. **应用 `take_first` 过滤器**:
   - 从所有重复响应中只取第一个：`["11"]`

4. **最终结果** (`filtered_resps`):

   ```json
   ["11"]
   ```

**不同过滤器的输出**：

- `score-first`: `["11"]` - 只使用第一个响应提取的答案
- `maj@64`: `["18"]` - 从 64 个响应中提取所有答案，然后多数投票得到最终答案
- `maj@8`: `["18"]` - 从前 8 个响应中提取答案，然后多数投票

**注意**：
- `filtered_resps` 的长度通常为 1（对应一个请求）
- 每个元素是字符串，表示提取/处理后的答案
- 如果过滤器无法提取答案，可能返回 `"[invalid]"` 或其他 fallback 值

#### 5. `filter` - 过滤器名称

**含义**：标识当前样本使用的过滤器管道（filter pipeline）名称。这个名称对应任务配置中 `filter_list` 里定义的过滤器名称。

**示例值**：
- `"score-first"`: 使用第一个响应，通过正则提取答案
  - 处理流程：`regex` → `take_first`
  - 结果：从第一个响应中提取答案

- `"maj@64"`: 使用所有 64 个响应的多数投票
  - 处理流程：`regex` → `majority_vote` → `take_first`
  - 结果：从 64 个响应中提取所有答案，多数投票得到最终答案

- `"maj@8"`: 使用前 8 个响应的多数投票
  - 处理流程：`take_first_k(k=8)` → `regex` → `majority_vote` → `take_first`
  - 结果：从前 8 个响应中提取答案，多数投票得到最终答案

**重要**：
- 每个样本的 `filter` 字段对应一个特定的过滤器管道
- 同一个样本可能有多个 `filter` 值（如果任务配置了多个过滤器），但每个样本记录只对应一个过滤器
- `filtered_resps` 中的值就是该过滤器处理后的结果

## 示例

### 完整示例（简化版）

```json
{
  "doc_id": 0,
  "doc": {
    "question": "Janet's ducks lay 16 eggs per day...",
    "answer": "Janet sells 16 - 3 - 4 = 9 duck eggs..."
  },
  "target": "18",
  "arguments": {
    "gen_args_0": {
      "arg_0": "Q: There are 15 trees...\nA: There are 15 trees...\n\nQ: Janet's ducks...\nA:",
      "arg_1": {
        "until": ["Q:", "\n\n"],
        "do_sample": true,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 1024
      }
    }
  },
  "resps": [
    ["response_1", "response_2", ..., "response_64"]
  ],
  "filtered_resps": ["11"],
  "filter": "score-first",
  "metrics": ["exact_match"],
  "doc_hash": "986c19252f3c7f6eaeff2229e9a5777f96a85c0ceea931a92d2289084dfd518c",
  "prompt_hash": "901fead9abd2535b6d0610c3c90125f38cad81678c4b2b98ec8fb818e3948bde",
  "target_hash": "4ec9599fc203d176a301536c2e091a19bc852759b255bd6818810a42c5fed14a",
  "exact_match": 0.0
}
```

## 文件位置

输出文件保存在 `output_path` 指定的目录下：

```
output_path/
  └── model_name_sanitized/
      ├── results_task_name_timestamp.json
      └── samples_task_name_timestamp.jsonl
```

例如：

```
./results/
  └── __datacenter__models__Qwen__Qwen2.5-0.5B/
      ├── results_gsm8k_cot_self_consistency_qwen3_4b_2026-01-28T03-22-54.json
      └── samples_gsm8k_cot_self_consistency_qwen3_4b_2026-01-28T03-22-54.jsonl
```

## 使用场景

1. **错误分析**: 查看模型在哪些样本上失败
2. **响应质量检查**: 查看模型的原始输出
3. **过滤器效果分析**: 比较不同过滤器的结果
4. **提示词调试**: 查看实际发送给模型的提示词
5. **后处理分析**: 对模型输出进行自定义分析

## 读取示例

### Python 读取 JSONL 文件

```python
import json

# 读取所有样本
samples = []
with open('samples_task_name_timestamp.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line.strip())
        samples.append(sample)

# 查看第一个样本
print(f"Doc ID: {samples[0]['doc_id']}")
print(f"Question: {samples[0]['doc']['question']}")
print(f"Target: {samples[0]['target']}")
print(f"Filtered Response: {samples[0]['filtered_resps']}")
print(f"Exact Match: {samples[0]['exact_match']}")
```

### 使用 HuggingFace datasets 读取

```python
from datasets import load_dataset

# 直接加载 JSONL 文件
dataset = load_dataset('json', data_files='samples_task_name_timestamp.jsonl', split='train')

# 查看数据
print(dataset[0])
```

## 注意事项

1. **文件大小**: `samples_*.jsonl` 文件可能非常大，特别是当 `repeats` 很大时
2. **内存使用**: 读取整个文件到内存时注意内存限制
3. **哈希值**: `doc_hash`, `prompt_hash`, `target_hash` 用于去重和缓存
4. **arguments 格式**: `arguments` 字段被重新格式化以便序列化，原始结构在 `arg_0`, `arg_1` 中
