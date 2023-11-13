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


class OctoAIEndpointRunnerBase():
  url_postfix = None
  reset = None

  def __init__(
      self,
      model_name: str,
      url: str=None,
      batch_size: int=1,
      top_p: float=0.0,
      temperature: float=0.0,
  ):
    """
    param model_name: str
        Model name from the list of models supported by OctoAI
    """
    self.model_name = model_name
    self.batch_size=batch_size
    self.url = url if url is not None else model_urls[self.model_name]

    self.init_msg_header()
    self.init_base_msg(top_p, temperature)

  def init_msg_header(self):
    # Get the API key from the environment variables
    api_key=os.environ["OCTOAI_API_KEY"]

    if api_key is None:
      raise ValueError("API_KEY not found in the .env file")

    self.headers = {
      "accept": "text/event-stream",
      "authorization": f"Bearer {api_key}",
      "content-type": "application/json",
    }

  def init_base_msg(self, top_p, temperature):
    self.base_msg = {
        "model": self.model_name,
        "stream": False,
        "max_tokens": 256,
        "top_p": top_p,
        "temperature": temperature,
    }

  def get_base_msg(self):
    return self.base_msg

  def call_octoai_reset(self):
    try:
      resp = requests.post(self.url + "/chat/reset", headers = self.headers)
      return resp.json()
    except Exception as e:
      print(f"Error resetting chat for endpoint {self.url}")
      print(e)
      return

  def call_octoai_inference(self):
    response = requests.post(self.url + self.url_postfix, headers=self.headers, json=self.msg)

    if response.status_code != 200:
      print(f"Error: {response.status_code} - {response.text}")

    return json.loads(response.text)

  def _batcher(self, requests):
    for i in range(0, len(requests), self.batch_size):
      yield requests[i:i + self.batch_size]

  def run(self, requests, results):
    if self.batch_size > 1:
      for request_batch in self._batcher(requests):
        try:
          # TODO(vvchernov): Use model_generate_parallel(...) when it becomes possible
          self.model_generate_batch(request_batch, results)
        except ConnectionError as e:
          print(f"ConnectionError: {e}. Skipping this batch and continuing...")
    else:
      for request in requests:
        try:
          self.model_generate(request, results)
        except ConnectionError as e:
          print(f"ConnectionError: {e}. Skipping this request and continuing...")

  def model_generate(self, request, results):
    success = False
    self.prepare_msg_data(request)
    for _ in range(REPEAT_REQUEST_TO_OCTOAI_SERVER):
      if self.reset:
        # TODO(vvchernov): process wrong reset
        self.call_octoai_reset()
      response = self.call_octoai_inference()
      if 'choices' in response.keys():
        success = True
        break
    if success:
      results.append(self.get_result(response))
    else:
      print("ERROR: responce does not have choices. Dummy response was inserted")
      results.append(self.dummy_result())

  def model_generate_batch(self, request_batch, results):
    parallel_results={}
    for id in range(len(request_batch)):
      parallel_results[id]=[]
      self.model_generate(request_batch[id], parallel_results[id])

    # Collect results together
    for id in range(len(request_batch)):
      results.extend(parallel_results[id])

  def model_generate_parallel(self, request_batch, results):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
      futures = []
      parallel_results={}
      for id in range(len(request_batch)):
        parallel_results[id]=[]
        futures.append(executor.submit(self.model_generate, request_batch[id], parallel_results[id]))

      for future in concurrent.futures.as_completed(futures):
        try:
          future.result()
        except Exception as exc:
          print(f"Error parallel generating predictions: {exc}")

      # Collect results together
      for id in range(len(request_batch)):
        results.extend(parallel_results[id])

  def get_result(self, response):
    raise NotImplementedError("get_result method is not implemented in base class")

  def dummy_result(self):
    raise NotImplementedError("dummy_result method is not implemented in base class")


class OctoAIEndpointRunnerGreedyUntil(OctoAIEndpointRunnerBase):
  url_postfix = "/v1/chat/completions"
  reset = True

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.msg = self.get_base_msg()

  def prepare_msg_data(self, request):
    inp = request[0]
    # TODO(vvchernov): use until to init stop tokens
    # request_args = request[1]
    # until = request_args["until"]
    self.msg["messages"] = [
        {
            "role": "user",
            "content": inp,
        }
    ]

  def get_result(self, response):
    return response['choices'][0]['message']['content']

  def dummy_result(self):
    return "Dummy response"


class OctoAIEndpointRunnerLogLikelihood(OctoAIEndpointRunnerBase):
  url_postfix = "/v1/logprob"
  reset = False

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.msg = self.get_base_msg()

  def prepare_msg_data(self, request):
    self.msg["context"] = request[0]
    self.msg["continuation"] = request[1]

  def get_result(self, response):
    logprob = response["logprob"]
    is_greedy = response["is_greedy"]
    return (logprob, is_greedy)

  def dummy_result(self):
    import sys
    return (-sys.float_info.max, False)


runners = {
  "greedy": OctoAIEndpointRunnerGreedyUntil,
  "loglikelihood": OctoAIEndpointRunnerLogLikelihood,
}

def get_octoai_runner(runner_name: str):
  if not runner_name in runners.keys():
    raise ValueError(f"{runner_name} is not a name of octoai runner")
  return runners[runner_name]


class OctoAIEndpointLM(BaseLM):
  def __init__(
      self,
      model_name: str="llama2-7b-chat-mlc-q0f16",
      url: str=None,
      batch_size: int=1,
      max_batch_size: int=None,
      device: str=None,
      top_p: float=0.0,
      temperature: float=0.0,):
    """
    :param model_name: str
        Model name from the list of models supported by OctoAI
    """
    super().__init__()

    self.time_meas = True

    # TODO(vvchernov): check that model name is supported or there is url
    self.model_name = model_name
    self._batch_size=int(batch_size)
    self.max_batch_size=max_batch_size
    self._device=device

    self.runner_args = {
      "model_name": self.model_name,
      "url": url,
      "batch_size": self._batch_size,
      "top_p": top_p,
      "temperature": temperature,
    }

  @property
  def eot_token_id(self):
    raise NotImplementedError("No eot token is supported.")

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

  def test(self, runner, requests):
    results = []
    if self.time_meas:
      start_timer = time.time()

    runner.run(requests, results)

    if self.time_meas:
      stop_timer = time.time()
      secs = stop_timer - start_timer
      print(
        "Full time of predictions measurement: {:.2f} sec, {:.2f} min, {:.2f} hour(s)".format(
            secs, secs / 60, secs / 3600))

    return results

  def loglikelihood(self, requests):
    if not requests:
      return []

    runner = get_octoai_runner("loglikelihood")(**self.runner_args)

    return self.test(runner, requests)

  def greedy_until(self, requests):
    if not requests:
      return []

    runner = get_octoai_runner("greedy")(**self.runner_args)

    return self.test(runner, requests)

  def _model_call(self, inps):
    raise NotImplementedError("OctoAI does not support one model call")
