import requests
import os
import json
import time

from lm_eval.base import BaseLM

REPEAT_REQUEST_TO_OCTOAI_SERVER = 10

model_urls = {
   "codellama-7b-instruct-mlc-q0f16": "https://codellama-7b-instruct-fp16-1gpu-g2ave3d5t9mm.octoai.run",
   "codellama-7b-instruct-mlc-q4f16_1": "https://codellama-7b-instruct-int4-1gpu-g2ave3d5t9mm.octoai.run",
   "codellama-7b-instruct-mlc-q8f16_1": "https://codellama-7b-instruct-int8-1gpu-g2ave3d5t9mm.octoai.run",
   "codellama-13b-instruct-mlc-q0f16": "https://codellama-13b-instruct-fp16-2gpu-g2ave3d5t9mm.octoai.run",
   "codellama-13b-instruct-mlc-q4f16_1": "https://codellama-13b-instruct-int4-1gpu-g2ave3d5t9mm.octoai.run",
   "codellama-13b-instruct-mlc-q8f16_1": "https://codellama-13b-instruct-int8-1gpu-g2ave3d5t9mm.octoai.run",
   "codellama-34b-instruct-mlc-q0f16": "https://codellama-34b-instruct-fp16-4gpu-g2ave3d5t9mm.octoai.run",
   "codellama-34b-instruct-mlc-q4f16_1": "https://codellama-34b-instruct-int4-1gpu-g2ave3d5t9mm.octoai.run",
   "codellama-34b-instruct-mlc-q8f16_1": "https://codellama-34b-instruct-int8-2gpu-g2ave3d5t9mm.octoai.run",
   "llama2-7b-chat-mlc-q0f16": "https://llama2-7b-chat-fp16-1gpu-g2ave3d5t9mm.octoai.run",
   "llama2-7b-chat-mlc-q4f16_1": "https://llama2-7b-chat-int4-1gpu-g2ave3d5t9mm.octoai.run",
   "llama2-7b-chat-mlc-q8f16_1": "https://llama2-7b-chat-int8-1gpu-g2ave3d5t9mm.octoai.run",
   "llama2-13b-chat-mlc-q0f16": "https://llama2-13b-chat-fp16-2gpu-g2ave3d5t9mm.octoai.run",
   "llama2-13b-chat-mlc-q4f16_1": "https://llama2-13b-chat-int4-1gpu-g2ave3d5t9mm.octoai.run",
   "llama2-13b-chat-mlc-q8f16_1": "https://llama2-13b-chat-int8-1gpu-g2ave3d5t9mm.octoai.run",
   "llama2-70b-chat-mlc-q0f16": "https://llama2-70b-chat-fp16-4gpu-g2ave3d5t9mm.octoai.run",
   "llama2-70b-chat-mlc-q4f16_1": "https://llama2-70b-chat-int4-2gpu-g2ave3d5t9mm.octoai.run",
   "llama2-70b-chat-mlc-q8f16_1": "https://llama2-70b-chat-int8-4gpu-g2ave3d5t9mm.octoai.run",
   # TODO(vvchernov): it is demo, may be need to remove
   "llama-2-70b-chat": "https://llama-2-70b-chat-demo-kk0powt97tmb.octoai.run",
}


class OctoAIEndpointLM(BaseLM):
  def __init__(
      self,
      model_name="llama2-7b-chat-mlc-q0f16",
      batch_size=1,
      max_batch_size=None,
      device=None):
    """
    :param model_name: str
        Model name from the list of models supported by OctoAI
    """
    super().__init__()

    self.time_meas = True

    self.model_name = model_name
    self._batch_size=int(batch_size)
    self.max_batch_size=max_batch_size
    self._device=device
    # TODO(vvchernov): check that model name is supported

    self.init_remote()

  def init_remote(self):
    # Get the API key from the environment variables
    api_key=os.environ["OCTOAI_API_KEY"]

    if api_key is None:
      raise ValueError("API_KEY not found in the .env file")

    self.url = model_urls[self.model_name]

    self.headers = {
      "accept": "text/event-stream",
      "authorization": f"Bearer {api_key}",
      "content-type": "application/json",
    }

    self.data = {
        "model": self.model_name,
        "messages": [
            {
                "role": "user",
                "content": "" # need to fill before use inference
            }
        ],
        "stream": False,
        "max_tokens": 256
    }

  @property
  def eot_token_id(self):
    raise NotImplementedError("No idea about anthropic tokenization.")

  @property
  def max_length(self):
    return 2048

  @property
  def max_gen_toks(self):
    return 256

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def device(self):
    return self._device

  def tok_encode(self, string: str):
      return string

  def tok_decode(self, tokens):
      return tokens

  def _loglikelihood_tokens(self, requests, disable_tqdm=False):
    raise NotImplementedError("No support for logits.")

  def greedy_until(self, requests):
    if not requests:
      return []

    results = []
    if self.time_meas:
      start_timer = time.time()
    if self.batch_size > 1:
      def _batcher(in_requests):
        for i in range(0, len(in_requests), self.batch_size):
          yield in_requests[i:i + self.batch_size]

      for request_batch in _batcher(requests):
        try:
          # TODO(vvchernov): Use _model_generate_parallel(...) when it becomes possible
          self._model_generate_batch(request_batch, results)
        except ConnectionError as e:
          print(f"ConnectionError: {e}. Skipping this batch and continuing...")

    else:
      for request in requests:
        inp = request[0]
        request_args = request[1]
        until = request_args["until"]
        try:
          self._model_generate(inp, results, stop=until)
        except ConnectionError as e:
          print(f"ConnectionError: {e}. Skipping this request and continuing...")

    if self.time_meas:
      stop_timer = time.time()
      secs = stop_timer - start_timer
      print(
        "Full time of predictions measurement: {:.2f} sec, {:.2f} min, {:.2f} hour(s)".format(
            secs, secs / 60, secs / 3600))

    return results

  def call_octoai_reset(self):
    try:
      resp = requests.post(self.url + "/chat/reset", headers = self.headers)
      return resp.json()
    except Exception as e:
      print(f"Error resetting chat for endpoint {self.url}")
      print(e)
      return

  def call_octoai_inference(self, user_input: str):
    self.data["messages"][0]["content"] = user_input
    response = requests.post(self.url + "/v1/chat/completions", headers=self.headers, json=self.data)

    if response.status_code != 200:
      print(f"Error: {response.status_code} - {response.text}")

    return response

  def _model_call(self, inps):
    raise NotImplementedError("OctoAI does not support one model call")

  # TODO(vvchernov): do we need additional args? max_tokens, temperature..
  def _model_generate(self, inps, results, stop=[]):
    success = False
    for _ in range(REPEAT_REQUEST_TO_OCTOAI_SERVER):
      # TODO(vvchernov): process wrong reset
      self.call_octoai_reset()
      response = self.call_octoai_inference(inps)
      response = json.loads(response.text)
      if 'choices' in response.keys():
        success = True
        break
    if success:
      results.append(response['choices'][0]['message']['content'])
    else:
      print("ERROR: responce does not have choices. Dummy response was inserted")
      results.append("Dummy response")

  def _model_generate_batch(self, request_batch, results):
    parallel_results={}
    for id in range(len(request_batch)):
      parallel_results[id]=[]
      inp = request_batch[id][0]
      request_args = request_batch[id][1]
      until = request_args["until"]
      self._model_generate(inp, parallel_results[id], stop=until)

    # Collect results together
    for id in range(len(request_batch)):
      results.extend(parallel_results[id])

  def _model_generate_parallel(self, request_batch, results):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
      futures = []
      parallel_results={}
      for id in range(len(request_batch)):
        parallel_results[id]=[]
        inp = request_batch[id][0]
        request_args = request_batch[id][1]
        until = request_args["until"]
        futures.append(executor.submit(self._model_generate, inp, parallel_results[id], stop=until))

      for future in concurrent.futures.as_completed(futures):
        try:
          future.result()
        except Exception as exc:
          print(f"Error parallel generating predictions: {exc}")

      # Collect results together
      for id in range(len(request_batch)):
        results.extend(parallel_results[id])
