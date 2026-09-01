import json
from pathlib import Path
from typing import Dict, List

import datasets


def all_keywords_included(output: str, keywords: List[str], ignore_case: bool = True) -> bool:
    """
    检查模型输出是否完整包含所有关键词。
    
    Args:
        output: 模型输出的字符串
        keywords: 需要检查的关键词列表
        ignore_case: 是否忽略大小写，默认 True
    
    Returns:
        True 如果所有关键词都在输出中，否则 False
    """
    if ignore_case:
        output_lower = output.lower()
        for keyword in keywords:
            if keyword.lower() not in output_lower:
                return False
    else:
        for keyword in keywords:
            if keyword not in output:
                return False
    return True


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """
    处理单个文档的结果并计算 metric。
    
    Args:
        doc: 文档字典，包含 expected_keywords 字段
        results: 模型生成的结果列表
    
    Returns:
        包含 metric 名称和分数的字典
    """
    output = results[0] if results else ""
    
    expected_keywords = doc.get("expected_keywords", [])
    
    if not expected_keywords:
        score = 1
    else:
        score = 1 if all_keywords_included(output, expected_keywords) else 0
    
    return {
        "keyword_acc": score,
    }


def load_local_jsonl(data_files: str = None, split: str = "train", **kwargs) -> datasets.DatasetDict:
    """
    加载本地 JSONL 文件作为数据集。
    
    Args:
        data_files: JSONL 文件的路径（可以是绝对路径或相对路径）
        split: 数据集拆分名称，默认 "train"
        **kwargs: 其他参数
    
    Returns:
        DatasetDict 对象，包含指定的拆分
    
    数据格式要求：
    每个 JSON 对象应包含：
    - prompt: 输入提示文本
    - expected_keywords: 期望关键词列表，例如 ["关键词1", "关键词2"]
    """
    if data_files is None:
        raise ValueError(
            "必须提供 data_files 参数指定 JSONL 文件路径。\n"
            "使用方式：在命令行中通过 --metadata='{\"data_files\": \"/path/to/your/data.jsonl\"}' 指定"
        )
    
    data_path = Path(data_files)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    dataset = datasets.load_dataset(
        "json",
        data_files=str(data_path),
        split=split,
    )
    
    return datasets.DatasetDict({split: dataset})
