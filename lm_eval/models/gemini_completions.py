import os
import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.models.openai_completions import LocalChatCompletion
import json

eval_logger = logging.getLogger(__name__)


@register_model("Gemini-completions")
class GeminiCompletionsAPI(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        
        self.kwargs = kwargs

        if "model_name" not in self.kwargs:
            raise ValueError(
                "GeminiCompletionsAPI requires a 'model_name' argument to be set."
            )
        self.model_name =  self.kwargs.pop("model_name")
        base_url = base_url.format(model=self.model_name)
        

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

        if "batch_size" in self.kwargs and int(self.kwargs["batch_size"]) > 1:
            eval_logger.warning(
                "Gemini API supports batching but not implemented yet, setting --batch_size=1" +
                ". The current approach uses --num_concurrent to send multiple requests."+
                ". If you want to use batching, please check: " + 
                "https://ai.google.dev/api/batch-api"
            )
            self._batch_size = 1

    @cached_property
    def eos_string(self) -> Optional[str]:
        return None #! NOT SURE (default value in API)

    @cached_property
    def api_key(self):
        key = os.environ.get("GEMINI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `GEMINI_API_KEY` environment variable."
            )
        return key

    @cached_property
    def header(self) -> dict:
        """Override this property to return the headers for the API request."""
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def create_message(
            self,
            messages: Any,  # Changed from List[str] as it's an object
            generate=False,
        ) -> List[dict]:
            """
            Creates a Gemini-compatible 'contents' list.
            
            This version is adapted to handle the incoming 'JsonChatStr' object,
            which contains an OpenAI-formatted chat history as a JSON string.
            """
            prefix = self.kwargs.get("prompt", '')

            # Check if messages is a simple list/tuple of strings (original logic)
            if messages and isinstance(messages[0], str):
                return [{
                    "role": "user",
                    "parts": [{"text": prefix + msg}]
                } for msg in messages]

            # --- Handle the JsonChatStr object ---
            try:
                # 1. Get the first item (the JsonChatStr object)
                msg_object = messages[0] 
                
                # 2. Get the 'prompt' attribute, which is a JSON string
                openai_json_str = msg_object.prompt 
                
                # 3. Parse the JSON string into a list of chat turns
                openai_history = json.loads(openai_json_str)
                
                gemini_contents = []
                prefix_applied = False
                
                # 4. Convert the OpenAI-formatted history to Gemini format
                for turn in openai_history:
                    role = turn.get("role")
                    content = turn.get("content")
                    
                    if not (role and content):
                        continue
                        
                    # Map OpenAI role "assistant" to Gemini "model"
                    if role == "assistant":
                        role = "model"
                    
                    # Prepend the global 'prefix' to the first user message
                    if role == "user" and not prefix_applied:
                        content = prefix + content
                        prefix_applied = True
                    
                    # Add the converted turn
                    gemini_contents.append({
                        "role": role,
                        "parts": [{"text": content}]
                    })
                # print(">>>>>>>>> gemini_contents:" , gemini_contents)
                return gemini_contents

            except Exception as e:
                # Fallback if the structure is not what we expect
                raise ValueError(
                    "Error processing message structure for Gemini API: {}".format(e)
                )

    
    def _create_payload(
        self,
        messages: List[dict],  # This is the 'contents' list from create_message
        generate=True,
        gen_kwargs: dict = None,
        seed: int = 1234,  # Note: Gemini API doesn't use a 'seed' parameter
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        
        # Handle system instruction
        system_instruction = ""
        if "system_instruction" in gen_kwargs:
            system_instruction = {
                "parts": [
                    {
                    "text": gen_kwargs.pop("system_instruction"),
                    }
                ]
            }
        
        # --- Handle Generation Parameters ---
        # 1. Max Tokens (OpenAI: max_tokens, Gemini: maxOutputTokens)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("maxOutputTokens", self._max_gen_toks)
        
        # 2. Temperature (Same name)
        temperature = gen_kwargs.pop("temperature", None)
        if temperature is None:
            temperature = 0
            eval_logger.debug("Temperature not specified, defaulting to 0.")
        
        # 3. Stop Sequences (OpenAI: stop, Gemini: stopSequences)
        stop = handle_stop_sequences(gen_kwargs.pop("until", []), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        if len(stop) > 4:
            eval_logger.info(
                "Gemini API supports up to 4 stop sequences. Truncating the list."
            )
            stop = stop[:4]

        # 4. 'n' (OpenAI: n, Gemini: candidateCount)
        candidateCount = gen_kwargs.pop("candidateCount", 1)
        if candidateCount != 1:
            raise NotImplementedError(
                "GeminiCompletionsAPI code (not the API itself) currently only supports candidateCount=1" +
                ". Please set candidateCount=1 or omit it from your generation parameters" +
                ". If you want to use candidateCount>1, please modify the parse_generations method accordingly."
            )
        if gen_kwargs.pop("do_sample", None) is not None:
            eval_logger.info(
                "Gemini API does not support a 'do_sample' parameter for generation. Ignored."
            )
        
        # 5. Handle thinkingBudget (specific to Gemini)
        thinking_budget = gen_kwargs.pop("thinking_budget", None) or gen_kwargs.pop("thinkingBudget", None)
        if thinking_budget is not None:
            if '2.5-flash' in self.model_name:
                eval_logger.info(
                    f"Using thinkingBudget={thinking_budget} in Gemini generationConfig."
                )
            elif '2.5-pro' in self.model_name:
                # eval_logger.warning(
                #     "Cannot adjust thinkingBudget for Gemini 2.5 Pro models. Delete thinkingBudget and try again."
                # )
                # thinking_budget = None
                raise ValueError(
                    "Cannot disable/adjust thinkingBudget for Gemini 2.5 Pro models. Remove thinkingBudget and try again. " +
                    "For more info check: https://ai.google.dev/gemini-api/docs/thinking"
                )
            else:
                eval_logger.info(
                    "thinkingBudget parameter is only applicable to Gemini 2.5 Flash Models. Ignored. " +
                    "you can find the full list of supported models here: https://ai.google.dev/gemini-api/docs/thinking"
                )
                thinking_budget = None
            
        # --- Create Gemini Payload ---
        
        # All generation parameters go inside 'generationConfig'
        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "stopSequences": stop,
            "candidateCount": candidateCount,
            **gen_kwargs,
        }
        if thinking_budget is not None:
            ThinkingConfig = {
                "thinkingBudget": thinking_budget
            }
            generation_config['thinkingConfig'] = ThinkingConfig
    
        # # Top P (OpenAI: top_p, Gemini: topP)
        # topP = gen_kwargs.pop("top_p", None) or gen_kwargs.pop("topP", None)
        # if topP:
        #     generation_config["topP"] = topP

        # # Top K (OpenAI: top_k, Gemini: topK)
        # topK = gen_kwargs.pop("top_k", None) or gen_kwargs.pop("topK", None)
        # if topK:
        #     generation_config["topK"] = topK
            
        final_configuration = {
            "contents": messages,
            "generationConfig": generation_config,
        }
        if system_instruction:
            final_configuration["systemInstruction"] = system_instruction
        return final_configuration
    
    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """
        Extracts the generated text from a Gemini API response.
        
        This simplified method assumes only one candidate is requested
        and just extracts the first available text.
        """
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]  # Ensure 'outputs' is always a list

        for out in outputs:
            try:
                # Since candidateCount=1, we just take the first candidate
                # at index 0 and the first text part at index 0.
                text = out["candidates"][0]["content"]["parts"][0]["text"]
                res.append(text)
            except (KeyError, IndexError, TypeError, AttributeError):
                # Handles cases where 'candidates', 'content', 'parts', 
                # or 'text' are missing, or if 'parts' is empty
                # (e.g., for a safety-blocked response).
                raise ValueError("Unexpected output format from Gemini API response. The recieved response is: {}".format(out))
                
        return res