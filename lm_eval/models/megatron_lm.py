"""
Megatron-LM backend for lm-evaluation-harness.

This module provides support for evaluating Megatron-LM models, including
both standard checkpoints and distributed checkpoints (.distcp format).

Requirements:
    - Megatron-LM must be installed or accessible via MEGATRON_PATH environment variable
    - PyTorch with CUDA support

Usage:
    # Set MEGATRON_PATH environment variable
    export MEGATRON_PATH=/path/to/Megatron-LM

    # Run evaluation
    lm-eval run --model megatron_lm \\
        --model_args load=/path/to/checkpoint,tokenizer_type=GPTSentencePieceTokenizer,tokenizer_model=/path/to/tokenizer.model \\
        --tasks hellaswag

    # With specific checkpoint step
    lm-eval run --model megatron_lm \\
        --model_args load=/path/to/checkpoint,ckpt_step=40000,tokenizer_model=/path/to/tokenizer.model \\
        --tasks hellaswag

    # With tensor parallelism
    lm-eval run --model megatron_lm \\
        --model_args load=/path/to/checkpoint,tokenizer_model=/path/to/tokenizer.model,tensor_model_parallel_size=2 \\
        --tasks hellaswag

    # With extra MCore arguments (e.g. --use-checkpoint-args, --no-rope-fusion, etc.)
    lm-eval run --model megatron_lm \\
        --model_args "load=/path/to/checkpoint,tokenizer_model=/path/to/tokenizer.model,extra_args=--use-checkpoint-args --no-rope-fusion --trust-remote-code --expert-tensor-parallel-size 1" \\
        --tasks hellaswag
"""

import logging
import os
import sys
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


eval_logger = logging.getLogger(__name__)


def _add_megatron_to_path():
    """Add Megatron-LM to Python path.

    The MEGATRON_PATH environment variable must be set to the Megatron-LM installation directory.
    """
    megatron_path = os.environ.get("MEGATRON_PATH")
    if megatron_path is None:
        raise EnvironmentError(
            "MEGATRON_PATH environment variable is not set. "
            "Please set it to the path of your Megatron-LM installation: "
            "export MEGATRON_PATH=/path/to/Megatron-LM"
        )
    if not os.path.isdir(megatron_path):
        raise FileNotFoundError(f"Megatron-LM directory not found at: {megatron_path}")
    if megatron_path not in sys.path:
        sys.path.insert(0, megatron_path)
    return megatron_path


def _check_dist_ckpt(load_path: str) -> bool:
    """检查是否是分布式检查点格式"""
    if not os.path.isdir(load_path):
        return False
    # 检查是否有 .distcp 文件
    for f in os.listdir(load_path):
        if f.endswith('.distcp'):
            return True
    # 检查是否有 metadata.json
    if os.path.exists(os.path.join(load_path, 'metadata.json')):
        return True
    return False


# def _resolve_checkpoint_step(load_path: str, ckpt_step: Optional[int] = None) -> str:
#     """
#     解析检查点路径，支持指定步数。
    
#     Args:
#         load_path: 检查点父目录（包含 iter_xxx 子目录）或直接的迭代目录
#         ckpt_step: 指定的检查点步数（如 40000），None 表示使用最新
        
#     Returns:
#         实际的检查点路径
#     """
#     # 如果路径本身就是 iter_xxx 格式，直接返回父目录
#     base_name = os.path.basename(load_path.rstrip('/'))
#     if base_name.startswith('iter_'):
#         return os.path.dirname(load_path)
    
#     # 检查是否有 iter_xxx 子目录
#     if not os.path.isdir(load_path):
#         return load_path
        
#     iter_dirs = [d for d in os.listdir(load_path) if d.startswith('iter_')]
#     if not iter_dirs:
#         return load_path
    
#     # 如果指定了 ckpt_step，查找对应的目录
#     if ckpt_step is not None:
#         target_dir = f"iter_{ckpt_step:07d}"
#         if target_dir in iter_dirs:
#             eval_logger.info(f"Using checkpoint step {ckpt_step}: {target_dir}")
#             return load_path  # 返回父目录，Megatron 会自己处理
#         else:
#             # 尝试不带前导零的格式
#             for iter_dir in iter_dirs:
#                 # 从 iter_0040000 提取数字
#                 try:
#                     step_num = int(iter_dir.split('_')[1])
#                     if step_num == ckpt_step:
#                         eval_logger.info(f"Using checkpoint step {ckpt_step}: {iter_dir}")
#                         return load_path
#                 except (IndexError, ValueError):
#                     continue
            
#             # 列出所有可用的检查点
#             available_steps = []
#             for iter_dir in sorted(iter_dirs):
#                 try:
#                     step_num = int(iter_dir.split('_')[1])
#                     available_steps.append(step_num)
#                 except (IndexError, ValueError):
#                     continue
            
#             raise ValueError(
#                 f"Checkpoint step {ckpt_step} not found. "
#                 f"Available steps: {available_steps}"
#             )
    
#     # 默认使用最新的检查点
#     latest_iter = sorted(iter_dirs)[-1]
#     eval_logger.info(f"Using latest checkpoint: {latest_iter}")
#     return load_path


def _parse_extra_args(extra_args: Optional[str]) -> List[str]:
    """
    解析 extra_args 字符串为命令行参数列表。
    
    使用空格分隔参数，支持 shell 风格的引号处理。
    
    Examples:
        "--no-rope-fusion --trust-remote-code" -> ["--no-rope-fusion", "--trust-remote-code"]
        "--expert-tensor-parallel-size 1 --no-rope-fusion" -> ["--expert-tensor-parallel-size", "1", "--no-rope-fusion"]
    """
    import shlex
    
    if not extra_args:
        return []
    
    try:
        return shlex.split(extra_args)
    except ValueError as e:
        eval_logger.warning(f"Failed to parse extra_args with shlex: {e}, falling back to simple split")
        return extra_args.split()


@register_model("megatron_lm")
class MegatronLMEval(LM):
    """
    Megatron-LM 模型适配器，用于 lm-evaluation-harness
    
    支持:
    - 标准 Megatron 检查点格式
    - 分布式检查点格式 (.distcp)
    
    Args:
        load: Megatron 检查点路径（父目录，包含 iter_xxx 子目录）
        ckpt_step: 指定加载的检查点步数 (如 40000 会加载 iter_0040000)，默认加载最新
        tokenizer_type: Tokenizer 类型 (如 GPTSentencePieceTokenizer, Qwen2Tokenizer)
        tokenizer_model: Tokenizer 模型文件路径
        vocab_file: 词表文件路径 (可选)
        merge_file: BPE merge 文件路径 (可选)
        tensor_model_parallel_size: 张量并行度
        pipeline_model_parallel_size: 流水线并行度
        seq_length: 最大序列长度
        micro_batch_size: 微批量大小 (可选，不指定则使用 checkpoint 中的值)
        max_gen_toks: 最大生成 token 数
        use_dist_ckpt: 是否使用分布式检查点格式 (自动检测)
        extra_args: 额外的 MCore 命令行参数，使用空格分隔
                    例如: "--use-checkpoint-args --no-rope-fusion --trust-remote-code --expert-tensor-parallel-size 1"
    """
    
    def __init__(
        self,
        load: str,
        ckpt_step: Optional[int] = None,
        tokenizer_type: str = "GPTSentencePieceTokenizer",
        tokenizer_model: Optional[str] = None,
        vocab_file: Optional[str] = None,
        merge_file: Optional[str] = None,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        seq_length: int = 4096,
        micro_batch_size: Optional[int] = None,
        max_gen_toks: int = 256,
        use_dist_ckpt: Optional[bool] = None,
        extra_args: Optional[str] = None,
        # 模型参数（如果不使用 --use-checkpoint-args）
        num_layers: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        num_query_groups: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        
        self._max_length = seq_length
        self._batch_size = micro_batch_size if micro_batch_size is not None else 1
        self._max_gen_toks = max_gen_toks
        self._load_path = load
        self._ckpt_step = ckpt_step
        
        # 添加 Megatron 到路径
        _add_megatron_to_path()
        
        # 处理检查点步数
        # actual_load_path = _resolve_checkpoint_step(load, ckpt_step)
        # self._actual_load_path = actual_load_path
        # eval_logger.info(f"Loading checkpoint from: {actual_load_path}")
        
        # 自动检测分布式检查点
        if use_dist_ckpt is None:
            # 检查迭代目录
            iter_dirs = [d for d in os.listdir(load) if d.startswith('iter_')]
            if iter_dirs:
                latest_iter = sorted(iter_dirs)[-1]
                iter_path = os.path.join(load, latest_iter)
                use_dist_ckpt = _check_dist_ckpt(iter_path)
            else:
                use_dist_ckpt = _check_dist_ckpt(load)
        
        self._use_dist_ckpt = use_dist_ckpt
        eval_logger.info(f"Using distributed checkpoint: {use_dist_ckpt}")
        
        # 初始化 Megatron 并加载模型
        self._initialize_megatron(
            load=load,
            ckpt_step=ckpt_step,
            tokenizer_type=tokenizer_type,
            tokenizer_model=tokenizer_model,
            vocab_file=vocab_file,
            merge_file=merge_file,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            use_dist_ckpt=use_dist_ckpt,
            extra_args=extra_args,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=ffn_hidden_size,
            num_query_groups=num_query_groups,
        )
        
        eval_logger.info(f"Megatron-LM model loaded from {load}")
        eval_logger.info(f"Max sequence length: {self._max_length}")
        eval_logger.info(f"Batch size: {self._batch_size}")

    def _initialize_megatron(self, **kwargs):
        """初始化 Megatron 分布式环境并加载模型"""
        from megatron.training import initialize_megatron, get_args, get_model, get_tokenizer
        from megatron.training.checkpointing import load_checkpoint
        from megatron.training.arguments import core_transformer_config_from_args
        
        # 构建命令行参数
        argv = [
            sys.argv[0],
            '--load', kwargs['load'],
            '--ckpt-step', str(kwargs['ckpt_step']),
            '--tensor-model-parallel-size', str(kwargs['tensor_model_parallel_size']),
            '--pipeline-model-parallel-size', str(kwargs['pipeline_model_parallel_size']),
            '--seq-length', str(kwargs['seq_length']),
            '--tokenizer-type', kwargs['tokenizer_type'],
            '--no-load-optim',
            '--no-load-rng',
            '--bf16',
            '--no-masked-softmax-fusion',
            '--no-bias-gelu-fusion',
            '--no-bias-dropout-fusion',
            '--no-async-tensor-model-parallel-allreduce',
            '--attention-softmax-in-fp32',
            '--use-cpu-initialization',
            '--exit-on-missing-checkpoint',
        ]
        
        if kwargs.get('micro_batch_size'):
            argv.extend(['--micro-batch-size', str(kwargs['micro_batch_size'])])
        
        if kwargs.get('use_dist_ckpt'):
            argv.append('--use-dist-ckpt')
            argv.append('--auto-detect-ckpt-format')
        
        if kwargs.get('tokenizer_model'):
            argv.extend(['--tokenizer-model', kwargs['tokenizer_model']])
        if kwargs.get('vocab_file'):
            argv.extend(['--vocab-file', kwargs['vocab_file']])
        if kwargs.get('merge_file'):
            argv.extend(['--merge-file', kwargs['merge_file']])
            
        # 如果需要手动指定模型参数
        if kwargs.get('num_layers'):
            argv.extend(['--num-layers', str(kwargs['num_layers'])])
        if kwargs.get('hidden_size'):
            argv.extend(['--hidden-size', str(kwargs['hidden_size'])])
        if kwargs.get('num_attention_heads'):
            argv.extend(['--num-attention-heads', str(kwargs['num_attention_heads'])])
        if kwargs.get('ffn_hidden_size'):
            argv.extend(['--ffn-hidden-size', str(kwargs['ffn_hidden_size'])])
        if kwargs.get('num_query_groups'):
            argv.extend(['--num-query-groups', str(kwargs['num_query_groups'])])
        
        # 添加额外的 MCore 参数
        extra_args_list = _parse_extra_args(kwargs.get('extra_args'))
        if extra_args_list:
            argv.extend(extra_args_list)
            eval_logger.info(f"Extra MCore args: {extra_args_list}")
        
        # 保存原始 argv 并替换
        original_argv = sys.argv
        sys.argv = argv
        
        eval_logger.info(f"Initializing Megatron with args: {' '.join(argv[1:])}")
        
        try:
            # 初始化 Megatron
            initialize_megatron(
                extra_args_provider=None,
                args_defaults={'tokenizer_type': kwargs['tokenizer_type']},
            )
            
            args = get_args()
            self._args = args
            
            # 如果用户指定了 micro_batch_size，使用用户指定的值；否则使用 args 中的值更新 self._batch_size
            if kwargs.get('micro_batch_size') is not None:
                requested_micro_batch_size = kwargs['micro_batch_size']
                if args.micro_batch_size != requested_micro_batch_size:
                    eval_logger.info(
                        f"Overriding micro_batch_size from checkpoint ({args.micro_batch_size}) "
                        f"to requested value ({requested_micro_batch_size})"
                    )
                    args.micro_batch_size = requested_micro_batch_size
            else:
                # 使用 checkpoint 或默认的 micro_batch_size
                self._batch_size = args.micro_batch_size
                eval_logger.info(f"Using micro_batch_size from args: {args.micro_batch_size}")
            
            # 获取 tokenizer
            self.tokenizer = get_tokenizer()
            
            # 创建 model_provider
            # 注意：Megatron-LM get_model 会传递 4 个参数: pre_process, post_process, config, pg_collection
            def model_provider(pre_process=True, post_process=True, config=None, pg_collection=None):
                """构建 GPT 模型"""
                from megatron.core.models.gpt import GPTModel
                from megatron.core.models.gpt.gpt_layer_specs import (
                    get_gpt_layer_local_spec,
                    get_gpt_layer_with_transformer_engine_spec,
                )
                
                # 如果没有传入 config，则从 args 获取
                if config is None:
                    config = core_transformer_config_from_args(args)
                
                # 选择 layer spec
                transformer_impl = getattr(args, 'transformer_impl', 'local')
                if transformer_impl == 'transformer_engine':
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        getattr(args, 'num_experts', None),
                        getattr(args, 'moe_grouped_gemm', False),
                        getattr(args, 'qk_layernorm', False),
                        getattr(args, 'multi_latent_attention', False),
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        getattr(args, 'num_experts', None),
                        getattr(args, 'moe_grouped_gemm', False),
                        getattr(args, 'qk_layernorm', False),
                        getattr(args, 'multi_latent_attention', False),
                    )
                
                model = GPTModel(
                    config=config,
                    transformer_layer_spec=transformer_layer_spec,
                    vocab_size=args.padded_vocab_size,
                    max_sequence_length=args.seq_length,
                    pre_process=pre_process,
                    post_process=post_process,
                    fp16_lm_cross_entropy=getattr(args, 'fp16_lm_cross_entropy', False),
                    parallel_output=False,
                    share_embeddings_and_output_weights=not getattr(args, 'untie_embeddings_and_output_weights', False),
                    position_embedding_type=getattr(args, 'position_embedding_type', 'learned_absolute'),
                    rotary_percent=getattr(args, 'rotary_percent', 1.0),
                    rotary_base=getattr(args, 'rotary_base', 10000),
                    seq_len_interpolation_factor=getattr(args, 'rotary_seq_len_interpolation_factor', None),
                )
                
                return model
            
            # 获取模型
            self._model = get_model(model_provider, wrap_with_ddp=False)
            
            # 加载检查点
            load_checkpoint(self._model, None, None, strict=True)
            
            # 提取单个模型（无虚拟流水线并行）
            assert len(self._model) == 1, f"Expected 1 model, got {len(self._model)}"
            self.model = self._model[0]
            self.model.eval()
            
            eval_logger.info("Model loaded successfully!")
            
        finally:
            sys.argv = original_argv
    
    @property
    def eot_token_id(self) -> int:
        """End of text token ID"""
        try:
            return self.tokenizer.eod
        except AttributeError:
            try:
                return self.tokenizer.eos_token_id
            except AttributeError:
                return self.tokenizer.eos_id
    
    @property
    def max_length(self) -> int:
        return self._max_length
    
    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    
    @property
    def rank(self) -> int:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    
    @property
    def world_size(self) -> int:
        return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    def tok_encode(self, string: str, add_special_tokens: bool = False) -> List[int]:
        """Tokenize 字符串"""
        try:
            return self.tokenizer.tokenize(string)
        except AttributeError:
            return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
    
    def tok_decode(self, tokens: List[int]) -> str:
        """Decode tokens 为字符串"""
        try:
            return self.tokenizer.detokenize(tokens)
        except AttributeError:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def _encode_pair(self, context: str, continuation: str) -> Tuple[List[int], List[int]]:
        """Encode context-continuation pair"""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        
        return context_enc, continuation_enc

    def _model_forward(
        self, 
        input_ids: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """模型前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # 创建 position_ids
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 创建 attention_mask (causal mask)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, 1, seq_len, seq_len), 
                dtype=torch.bool, 
                device=input_ids.device
            ).tril()
        
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        
        return output
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """计算 log-likelihood"""
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                context_enc = [self.eot_token_id]
                continuation_enc = self.tok_encode(continuation)
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
            new_reqs.append(((context, continuation), context_enc, continuation_enc))
        
        return self._loglikelihood_tokens(new_reqs)
    
    def _loglikelihood_tokens(
        self, 
        requests: List[Tuple],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """基于 tokens 计算 log-likelihood"""
        res = []
        
        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)
        
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(n=self.batch_size, batch_fn=None)
        
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm or (self.rank != 0),
            desc="Running loglikelihood requests",
        )
        
        for chunk in chunks:
            inps = []
            ctxlens = []
            contlens = []
            
            for _, context_enc, continuation_enc in chunk:
                # 截断到最大长度
                inp = (context_enc + continuation_enc)[-(self.max_length):]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - self.max_length
                )
                ctxlens.append(ctxlen)
                contlens.append(len(continuation_enc))
                inps.append(inp)
            
            # Pad sequences
            max_len = max(len(inp) for inp in inps)
            padded_inps = []
            for inp in inps:
                padded = [self.eot_token_id] * (max_len - len(inp)) + inp
                padded_inps.append(padded)
            
            input_ids = torch.tensor(padded_inps, dtype=torch.long, device=self.device)
            
            # Forward pass
            logits = self._model_forward(input_ids)
            
            # 计算 log probabilities
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            
            for i, (ctxlen, contlen) in enumerate(zip(ctxlens, contlens)):
                # 获取 padding 长度
                pad_len = max_len - len(inps[i])
                
                # 计算 continuation 的 log probability
                cont_log_probs = []
                greedy_tokens = []
                
                start_idx = pad_len + ctxlen - 1
                end_idx = pad_len + ctxlen + contlen - 1
                
                for j in range(start_idx, end_idx):
                    next_token = input_ids[i, j + 1].item()
                    cont_log_probs.append(log_probs[i, j, next_token].item())
                    greedy_tokens.append(torch.argmax(log_probs[i, j]).item())
                
                logprob = sum(cont_log_probs)
                
                # 检查是否 greedy
                actual_tokens = input_ids[i, start_idx + 1:end_idx + 1].cpu().tolist()
                is_greedy = greedy_tokens == actual_tokens
                
                answer = (logprob, is_greedy)
                res.append(answer)
                
                cache_key = chunk[i][0]
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                
                pbar.update(1)
        
        pbar.close()
        return re_ord.get_original(res)
    
    def loglikelihood_rolling(
        self, 
        requests: List[Instance],
        disable_tqdm: bool = False,
    ) -> List[float]:
        """计算 rolling log-likelihood (用于 perplexity)"""
        loglikelihoods = []
        
        for (string,) in tqdm(
            [req.args for req in requests], 
            disable=disable_tqdm or (self.rank != 0),
            desc="Running loglikelihood_rolling requests",
        ):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )
            
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]
            string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)
            string_nll = [x[0] for x in string_nll]
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
            
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)
        
        return loglikelihoods
    
    def generate_until(
        self, 
        requests: List[Instance],
        disable_tqdm: bool = False,
    ) -> List[str]:
        """生成文本直到停止条件"""
        results = []
        
        for request in tqdm(
            requests, 
            disable=disable_tqdm or (self.rank != 0),
            desc="Running generate_until requests",
        ):
            context, gen_kwargs = request.args
            gen_kwargs = deepcopy(gen_kwargs)
            
            until = gen_kwargs.pop("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.pop("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0.0)
            top_p = gen_kwargs.pop("top_p", 1.0)
            top_k = gen_kwargs.pop("top_k", 0)
            
            # Tokenize context
            context_tokens = self.tok_encode(context)
            context_tokens = context_tokens[-(self.max_length - max_gen_toks):]
            
            input_ids = torch.tensor([context_tokens], dtype=torch.long, device=self.device)
            generated_tokens = []
            
            # Autoregressive generation
            for _ in range(max_gen_toks):
                # 截断输入如果太长
                if input_ids.shape[1] > self.max_length:
                    input_ids = input_ids[:, -self.max_length:]
                
                logits = self._model_forward(input_ids)
                next_token_logits = logits[:, -1, :].float()
                
                # Sampling
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    
                    if top_k > 0:
                        top_k_vals, _ = torch.topk(next_token_logits, top_k)
                        threshold = top_k_vals[:, -1].unsqueeze(-1)
                        next_token_logits = torch.where(
                            next_token_logits < threshold,
                            torch.full_like(next_token_logits, float('-inf')),
                            next_token_logits
                        )
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = False
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                    
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                next_token_id = next_token.item()
                generated_tokens.append(next_token_id)
                
                # 检查 EOS
                if next_token_id == self.eot_token_id:
                    break
                
                # 更新输入
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # 检查停止序列
                generated_text = self.tok_decode(generated_tokens)
                should_stop = False
                for stop_seq in until:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        should_stop = True
                        break
                if should_stop:
                    break
            
            # 最终处理
            continuation = self.tok_decode(generated_tokens)
            for stop_seq in until:
                if stop_seq in continuation:
                    continuation = continuation.split(stop_seq)[0]
            
            results.append(continuation)
            self.cache_hook.add_partial("generate_until", request.args, continuation)
        
        return results
