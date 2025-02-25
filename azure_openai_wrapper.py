import logging
from typing import Dict, Any
from lm_eval.api.model import LM  # Import the base class
from openai import AzureOpenAI  # Import Azure OpenAI client

# Custom wrapper for Azure OpenAI
class AzureOpenAIWrapper(LM):
    def __init__(self, azure_openai_api_key: str, azure_openai_endpoint: str, deployment_name: str, **kwargs):
        super().__init__()
        self.client = AzureOpenAI(
            api_key=azure_openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=azure_openai_endpoint
        )

        self.deployment_name = deployment_name
        self.logger = logging.getLogger(__name__)
        self.raw_responses = []  # Store raw responses globally

    def loglikelihood(self, requests):
        """
        Compute the log-likelihood of a batch of requests with direct answers (no few-shot)
        """
        responses = []
        self.raw_responses = []  # Reset raw responses for this batch
        
        for request in requests:
            context, continuation = request.args
            try:
                # Extract just the current question, regardless of any accumulated context
                # We only want the last question if there are multiple
                last_question_idx = context.rfind("Question:")
                if last_question_idx != -1:
                    current_question = context[last_question_idx:]
                else:
                    current_question = context
                
                # Create a clear, direct prompt
                direct_prompt = f"""Answer the following multiple choice question with ONLY the letter of the correct answer (a, b, c, or d):

{current_question}"""
                
                # Strong system prompt for consistency
                system_prompt = "You are a legal expert. Answer ONLY with the single letter (a, b, c, or d) of the correct option. Do not include any explanations or additional text."
                
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": direct_prompt}
                    ],
                    max_tokens=5,  # Very limited to force brevity
                    temperature=0.0
                )
                
                # Get raw response content
                raw_completion = response.choices[0].message.content.strip()
                self.raw_responses.append(raw_completion)
                
                # Log the raw response
                self.logger.info(f"Raw response: {raw_completion}")
                
                # Check if the response contains one of the expected options
                option_markers = ['a', 'b', 'c', 'd']
                response_option = None
                
                # First, check if the response is just a single letter
                if len(raw_completion) == 1 and raw_completion.lower() in option_markers:
                    response_option = raw_completion.lower()
                else:
                    # Extract the first letter that matches an option
                    for char in raw_completion.lower():
                        if char in option_markers:
                            response_option = char
                            break
                
                # If found, check if it matches the expected option in continuation
                expected_option = None
                for marker in option_markers:
                    if marker == continuation.strip().lower():
                        expected_option = marker
                        break
                
                if response_option and expected_option:
                    is_correct = response_option == expected_option
                    confidence = 0.8 if is_correct else -0.8
                    self.logger.info(f"Extracted option: {response_option}, Expected: {expected_option}, Correct: {is_correct}")
                else:
                    # No valid option found or couldn't determine expected option
                    confidence = -1.0
                    is_correct = False
                    self.logger.warning(f"Could not extract a valid option from '{raw_completion}' or determine expected option")
                
                responses.append((confidence, is_correct))
                
            except Exception as e:
                self.logger.error(f"Error computing loglikelihood: {e}")
                responses.append((None, False))
                self.raw_responses.append(None)
                
        return responses

    def generate(self, requests):
        """
        Generate text for a batch of requests with direct answers (no few-shot)
        """
        responses = []
        for request in requests:
            prompt = request.args[0]
            try:
                # Extract just the current question, regardless of any accumulated context
                last_question_idx = prompt.rfind("Question:")
                if last_question_idx != -1:
                    current_question = prompt[last_question_idx:]
                else:
                    current_question = prompt
                
                # Create a clear, direct prompt
                direct_prompt = f"""Answer the following multiple choice question with ONLY the letter of the correct answer (a, b, c, or d):

{current_question}"""
                
                # Strong system prompt for consistency
                system_prompt = "You are a legal expert. Answer ONLY with the single letter (a, b, c, or d) of the correct option. Do not include any explanations or additional text."
                
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": direct_prompt}
                    ],
                    max_tokens=5,
                    temperature=0.0
                )
                
                responses.append(response.choices[0].message.content.strip())
            except Exception as e:
                self.logger.error(f"Error generating text: {e}")
                responses.append("")
        return responses

    def generate_until(self, requests):
        """
        Generate text until a stopping condition is met, with direct answers (no few-shot)
        """
        responses = []
        for request in requests:
            prompt = request.args[0]
            stop_sequences = ["\n"]
            if len(request.args) > 1:
                stop_sequences = request.args[1] if isinstance(request.args[1], list) else [request.args[1]]
            
            try:
                # Extract just the current question, regardless of any accumulated context
                last_question_idx = prompt.rfind("Question:")
                if last_question_idx != -1:
                    current_question = prompt[last_question_idx:]
                else:
                    current_question = prompt
                
                # Create a clear, direct prompt
                direct_prompt = f"""Answer the following multiple choice question with ONLY the letter of the correct answer (a, b, c, or d):

{current_question}"""
                
                # Strong system prompt for consistency
                system_prompt = "You are a legal expert. Answer ONLY with the single letter (a, b, c, or d) of the correct option. Do not include any explanations or additional text."
                
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": direct_prompt}
                    ],
                    max_tokens=5,
                    temperature=0.0,
                    stop=stop_sequences
                )
                
                responses.append(response.choices[0].message.content.strip())
            except Exception as e:
                self.logger.error(f"Error generating text until stop sequence: {e}")
                responses.append("")
        return responses

    def loglikelihood_rolling(self, requests):
        """
        Compute the log-likelihood of a rolling batch of requests.
        """
        raise NotImplementedError("loglikelihood_rolling is not implemented for Azure OpenAI")