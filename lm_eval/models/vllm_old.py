import inspect
import random
from lm_eval.models.huggingface import AutoCausalLM, TokenSequence, stop_sequences_criteria
import lm_eval.utils as utils
from vllm import LLM, SamplingParams
import torch
from typing import Optional, Union, List, Tuple
from tqdm import tqdm
import transformers

class VLLM(AutoCausalLM):
    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        batch_size: Optional[int] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        use_accelerate: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[int, str]] = "cuda",
    ):
        super().__init__(
            pretrained=pretrained,
            tokenizer=tokenizer,
            subfolder=subfolder,
            revision=revision,
            batch_size=batch_size,
            max_gen_toks=max_gen_toks,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            use_accelerate=use_accelerate,
            device_map_option=device_map_option,
            max_memory_per_gpu=max_memory_per_gpu,
            max_cpu_memory=max_cpu_memory,
            offload_folder=offload_folder,
            dtype=dtype,
            device=device,
        )
        self.llm = LLM(model=pretrained)
    
    # def greedy_until(self, requests):
    #     prompts = []
    #     stop = '\n\n'
    #     for request in requests:
    #         context, until, is_greedy, _model_generate_kwargs = self.parse_request(request)
    #         prompts.append(context)
    #         stop = until
    #     sampling_params = SamplingParams(temperature=0.0, top_p=1.0, stop=stop)
    #     outputs = self.llm.generate(prompts, sampling_params)

    #     res = [
    #         output.outputs[0].text
    #         for output in outputs
    #     ]

    #     return res

    def greedy_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        # Since we use VLLM, use unlimited batch size.
        chunk = reorder.get_reordered()
        context = [c[0] for c in chunk]
        until = chunk[0][1]
        for c in chunk:
            assert c[1] == until, \
                "`until` condition must be the same across batch elements (%s vs %s). Use batch_size=1." % (
                    c[1], until
                )
        max_tokens = self.max_gen_toks
        token_context = self.tok_encode_batch(context)
        generated_tokens = self._model_generate(
            inputs=token_context,
            max_tokens=max_tokens,
            stop=until,
        )
        generated_texts = self.postprocess(
            generated_tokens=generated_tokens,
            prefix_length=token_context['input_ids'].size(1),
            until=until,
            is_greedy=True
        )
        for text in generated_texts:
            self.cache_hook.add_partial("greedy_until", (context, until), text)
        results.extend(generated_texts)

        return reorder.get_original(results)

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        num_return_sequences: int = 1,
        num_return_sequences_batch: int = -1,
        temperature: float = 0.0
    ) -> TokenSequence:

        if isinstance(stop, str):
            stop = [stop]

        input_ids = inputs["input_ids"][:, self.max_gen_toks-self.max_length:]

        # We only support batching over examples when `num_return_sequences is 1`
        # (see todo in `.generate()`.
        assert (input_ids.size(0) == 1 or num_return_sequences == 1)

        input_ids = input_ids.numpy().tolist()
        bsz = len(input_ids)

        if num_return_sequences_batch > 0:
            print(f"Batching over {num_return_sequences_batch} sequences.")
            num_batches = num_return_sequences // num_return_sequences_batch
            num_return_sequences = num_return_sequences_batch
        else:
            num_batches = 1

        generated_tokens = []
        for _ in range(num_batches):
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=stop, n=num_return_sequences)
            outputs = self.llm.generate(prompt_token_ids=input_ids, 
                                        sampling_params=sampling_params,
                                        use_tqdm=bsz > 1)
            output_tokens = []
            # Sort by request_id
            outputs = sorted(outputs, key=lambda x: x.request_id)
            for output in outputs:
                prefix_tokens = output.prompt_token_ids
                for gen in output.outputs:
                    gen_tokens = gen.token_ids
                    output_tokens.append(prefix_tokens + gen_tokens)
            generated_tokens.extend(output_tokens)
        # Make each element of the list into a torch tensor
        generated_tokens = [torch.tensor(t).view(1, -1) for t in generated_tokens]
        generated_tokens = self._pad_and_combine(generated_tokens)
        return generated_tokens

    def generate(self, requests):
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        # TODO: batching along example dimension (in addition to the current batching
        # along the `num_return_sequences` dimension). This would involve
        # iterating over chunks (`utils.chunks`) and modifying `._model_generate`.
        for request in tqdm(re_ord.get_reordered()):
            context, until, is_greedy, _model_generate_kwargs = self.parse_request(request)

            context_enc = self.tok_encode_batch(context)
            generated_tokens = self._model_generate(
                inputs=context_enc,
                max_tokens=self.max_gen_toks,
                stop=until,
                **_model_generate_kwargs
            )
            generated_texts = self.postprocess(
                generated_tokens=generated_tokens,
                prefix_length=context_enc['input_ids'].shape[1],
                until=until,
                is_greedy=is_greedy
            )
            cache = (context, until, tuple(_model_generate_kwargs))
            self.cache_hook.add_partial("generate", cache, generated_texts)
            res.append(generated_texts)
        return re_ord.get_original(res)